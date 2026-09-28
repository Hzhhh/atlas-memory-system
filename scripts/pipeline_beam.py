"""BEAM answer generation and rubric evaluation aligned to the upstream protocol."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_config import (ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL, JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL, JUDGE_VERSION)


UPSTREAM_REPOSITORY = "mohammadtavakoli78/BEAM"
UPSTREAM_COMMIT = "3e12035532eb85768f1a7cd779832b650c4b2ef9"
DEFAULT_MODEL = "Qwen/Qwen3-14B"

# Verbatim src/prompts.py:answer_generation_for_rag at UPSTREAM_COMMIT.
ANSWER_GENERATION_FOR_RAG = """
You are an assistant that MUST answer questions using ONLY the information provided in the context below. 

STRICT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Do NOT use your internal knowledge

CONTEXT:
<context>

QUESTION:
<question>

ANSWER REQUIREMENTS:
- Be direct and concise
- Only output the answer to the question without any explanation 

RESPONSE:
"""

# Verbatim src/prompts.py:unified_llm_judge_base_prompt at UPSTREAM_COMMIT.
UNIFIED_LLM_JUDGE_BASE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. 

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.
  
- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.
  
- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)
   
3. **Check Compliance**: 
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""


BATCH_OUTPUT_FORMAT = """## OUTPUT FORMAT:
Return one independent evaluation for every indexed rubric criterion in JSON:

{
  "scores": [
    {"index": 0, "score": 1.0, "reason": "detailed justification"}
  ]
}

Include every index exactly once. Each score must be 1.0, 0.5, or 0.0.
NOTE: ONLY output the json object, without any explanation before or after that
"""


EQUIVALENCE_SYSTEM_PROMPT = """You are a binary classifier.
If the TWO snippets describe the SAME event/fact, reply **YES**
Otherwise reply **NO**. No extra words.
DO NOT provide any exaplanation."""


import argparse
import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_config import (ANSWER_API_BASE, ANSWER_API_KEY, ANSWER_MODEL, JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL, JUDGE_VERSION)

CONCURRENCY = int(os.environ.get("PIPELINE_CONCURRENCY", "8"))


def rows(path):
    # 只按 \n 分行: splitlines() 会按 U+2028 等行边界字符多拆行导致 JSON 截断
    text = Path(path).read_text(encoding="utf-8")
    parsed = [json.loads(line) for line in text.split("\n") if line.strip()]
    ids = [row.get("id") for row in parsed]
    if any(i is None for i in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path} must contain unique, non-empty ids")
    return parsed


def text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def context_text(item):
    for key in ("context", "retrieved_context", "memories"):
        if key in item:
            return text(item[key])
    speaker_sections = []
    for number in (1, 2):
        key = f"speaker_{number}_memories"
        if key in item:
            name = text(item.get(f"speaker_{number}_name", f"speaker {number}"))
            speaker_sections.append(f"Memories for user {name}:\n{text(item[key])}")
    if speaker_sections:
        return "\n\n".join(speaker_sections)
    raise ValueError(f"record {item.get('id', '<unknown>')} has no context or memories")


def render_answer_prompt(item):
    values = {"context": context_text(item), "question": text(item["question"])}
    return re.sub("<(context|question)>", lambda m: values[m.group(1)], ANSWER_GENERATION_FOR_RAG)


def rubric_items(item):
    value = item.get("rubric_nuggets", item.get("rubrics", item.get("rubric")))
    if not isinstance(value, list) or not value:
        raise ValueError(f"record {item.get('id', '<unknown>')} has no rubric list")
    result = []
    for rubric in value:
        if isinstance(rubric, dict):
            rubric = rubric.get("rubric_criteria", rubric.get("criterion", rubric.get("text")))
        if not isinstance(rubric, str) or not rubric.strip():
            raise ValueError("invalid rubric")
        result.append(rubric.strip())
    return result


def render_batch_judge_prompt(question, response, rubrics):
    criteria = "\n".join(f"[{i}] {r}" for i, r in enumerate(rubrics))
    prompt = (UNIFIED_LLM_JUDGE_BASE_PROMPT
              .replace("<question>", question)
              .replace("<rubric_item>", criteria)
              .replace("<llm_response>", response))
    prompt = prompt[:prompt.index("## OUTPUT FORMAT:")] + BATCH_OUTPUT_FORMAT
    return ("Evaluate every indexed RUBRIC CRITERION independently. Apply the complete protocol "
            "below separately to each criterion; do not let one criterion affect another.\n\n" + prompt)


def parse_json_object(response):
    candidate = response.strip()
    if candidate.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            raise ValueError("no JSON object in response") from None
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    return payload


def parse_rubric_scores(response, count):
    payload = parse_json_object(response)
    raw = payload.get("scores")
    if not isinstance(raw, list):
        raise ValueError("missing scores list")
    scores = {}
    for item in raw:
        index = int(item["index"])
        score = float(item["score"])
        if index in scores or not 0 <= index < count:
            raise ValueError("bad index")
        if score not in {0.0, 0.5, 1.0}:
            raise ValueError("score must be 0/0.5/1")
        scores[index] = {"index": index, "score": score, "reason": str(item.get("reason", "")).strip()}
    if set(scores) != set(range(count)):
        raise ValueError("must return every index once")
    return [scores[i] for i in range(count)]


async def call_model(client, args, messages, max_tokens, json_mode=False):
    payload = {"model": args.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if json_mode and os.environ.get("PIPELINE_JSON_MODE", "1") == "1":
        payload["response_format"] = {"type": "json_object"}
    # 阿里云 MaaS 兼容模式: 非流式调用 qwen3 必须显式关思考(顶层参数)
    if os.environ.get("PIPELINE_ALI_MODE", "0") == "1":
        payload["enable_thinking"] = False
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    last = None
    for attempt in range(6):
        try:
            resp = await client.post(endpoint, headers=headers, json=payload)
            if resp.status_code == 429 and attempt < 5:
                await asyncio.sleep(min(90, 12 * (attempt + 1)))
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            last = exc
            if attempt == 5:
                raise
            await asyncio.sleep(min(60, 3 * (2 ** attempt)))
    raise RuntimeError(f"unreachable: {last}")


async def snippets_equivalent(client, args, reference, candidate):
    response = await call_model(client, args, [
        {"role": "system", "content": EQUIVALENCE_SYSTEM_PROMPT},
        {"role": "user", "content": f"First snippet: {reference}\n\nSecond snippet: {candidate}"},
    ], 8)
    return "yes" in response.casefold()


async def align_with_llm(client, args, reference, system):
    used = set()
    out = []
    for candidate in system:
        matched = None
        for index, expected in enumerate(reference):
            if index in used:
                continue
            if await snippets_equivalent(client, args, expected, candidate):
                matched = index
                break
        if matched is None:
            out.append(candidate)
        else:
            out.append(reference[matched])
            used.add(matched)
    return reference, out


def kendall_tau_b(first, second):
    conc = disc = t1 = t2 = 0
    for l in range(len(first)):
        for r in range(l + 1, len(first)):
            d1, d2 = first[l] - first[r], second[l] - second[r]
            if d1 == 0 and d2 == 0:
                continue
            if d1 == 0:
                t1 += 1
            elif d2 == 0:
                t2 += 1
            elif d1 * d2 > 0:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + t1) * (conc + disc + t2))
    return (conc - disc) / denom if denom else 0.0


def event_ordering_metrics(reference, system):
    tp = len(set(reference) & set(system))
    fp = len([x for x in system if x not in reference])
    fn = len([x for x in reference if x not in system])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = list(dict.fromkeys(reference + system))
    tie = len(union) + 1

    def ranks(seq):
        pos = {x: i + 1 for i, x in enumerate(seq)}
        return [pos.get(x, tie) for x in union]

    tau = (kendall_tau_b(ranks(reference), ranks(system)) + 1) / 2
    return {"precision": precision, "recall": recall, "f1": f1, "tau_norm": tau, "final_score": tau * f1}


async def answer(args):
    args.api_key = ANSWER_API_KEY
    args.base_url = ANSWER_API_BASE
    args.model = ANSWER_MODEL
    items = rows(args.input)
    output = Path(args.output)
    done = {i["id"] for i in rows(output)} if output.exists() else set()
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=600) as client:
        with output.open("a", encoding="utf-8") as handle:

            async def one(item):
                try:
                    async with sem:
                        gen = await call_model(client, args,
                                               [{"role": "user", "content": render_answer_prompt(item)}],
                                               args.max_tokens)
                    return {"id": item["id"], "generated_answer": gen}
                except Exception as exc:
                    return {"id": item["id"], "generated_answer": "", "error": str(exc)[:150]}

            futures = [asyncio.create_task(one(i)) for i in items if i["id"] not in done]
            for fut in asyncio.as_completed(futures):
                handle.write(json.dumps(await fut, ensure_ascii=False) + "\n")
                handle.flush()


async def evaluate(args):
    args.api_key = JUDGE_API_KEY
    args.base_url = JUDGE_API_BASE
    args.model = JUDGE_MODEL
    items = {i["id"]: i for i in rows(args.input)}
    answers = {i["id"]: i["generated_answer"] for i in rows(args.answers)}
    if set(items) != set(answers):
        raise SystemExit("input/answer ID mismatch")
    output = Path(args.output)
    done = {i["id"] for i in rows(output) if "llm_judge_score" in i} if output.exists() else set()
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=600) as client:
        with output.open("a", encoding="utf-8") as handle:

            async def one(ident):
                item = items[ident]
                try:
                    rubrics = rubric_items(item)
                    jr = await call_model(client, args, [
                        {"role": "user",
                         "content": render_batch_judge_prompt(text(item["question"]), answers[ident], rubrics)}],
                        args.judge_max_tokens, json_mode=True)
                    scores = parse_rubric_scores(jr, len(rubrics))
                    result = {
                        "id": ident,
                        "question_type": item.get("question_type", item.get("category")),
                        "judge_model": args.model,
                        "llm_judge_score": sum(s["score"] for s in scores) / len(scores),
                        "rubric_scores": [{"rubric": r, "score": s["score"]} for r, s in zip(rubrics, scores)],
                    }
                    if result["question_type"] == "event_ordering":
                        reference, system = await align_with_llm(client, args, rubrics, answers[ident].split("\n"))
                        result["event_ordering"] = event_ordering_metrics(reference, system)
                    return result
                except Exception as exc:
                    return {"id": ident, "llm_judge_score": 0.0, "error": str(exc)[:150]}

            futures = [asyncio.create_task(one(i)) for i in items if i not in done]
            for fut in asyncio.as_completed(futures):
                handle.write(json.dumps(await fut, ensure_ascii=False) + "\n")
                handle.flush()


def parser():
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    ap = commands.add_parser("answer")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.set_defaults(run=answer)
    ep = commands.add_parser("evaluate")
    ep.add_argument("--input", required=True)
    ep.add_argument("--answers", required=True)
    ep.add_argument("--output", required=True)
    ep.add_argument("--model", default=DEFAULT_MODEL)
    ep.add_argument("--base-url", default=None)
    ep.add_argument("--api-key-env", default=None)
    ep.add_argument("--judge-max-tokens", type=int, default=1024)
    ep.set_defaults(run=evaluate)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    asyncio.run(arguments.run(arguments))
