# -*- coding: utf-8 -*-
"""EvRecall：gold 证据词在检索上下文中的覆盖率（零 LLM，检索层优化靶心）。

用法:
  python scripts/evrecall.py outputs/eval_input_beam_100k_message.jsonl [more.jsonl ...]
  python scripts/evrecall.py --all          # 跑五集主基线

口径: gold_answer 实词(>3字符,去停词) 在 retrieved_context 中的覆盖率, 按题平均。
注意: 这是证据在场的近似度量——开放域题 gold 是要点式答案, 覆盖率语义仍成立。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # aml-system 根

STOP = set("""the a an and or but if then else of to in on at by for with about as is are was were be been being
do does did done have has had having will would can could should may might must shall this that these those there
here it its from not no nor so than too very just also only own same such then once
i you he she we they me him her us them my your his their our
what which who whom whose when where why how
say says said get got go goes went make makes made take takes took know knows knew
""".split())

_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]+")


def content_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(str(text)) if len(w) > 3 and w.lower() not in STOP]


def evidence_recall(gold: str, context: str) -> tuple[float, int]:
    """返回 (覆盖率, gold 实词数)。"""
    words = content_words(gold)
    if not words:
        return 1.0, 0
    ctx = set(content_words(context))
    hit = sum(1 for w in words if w in ctx)
    return hit / len(words), len(words)


def load_rows(path: str) -> list[dict]:
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(ROOT, path)  # 相对路径锚到仓库根
    rows = []
    with open(path, encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cat_of(row: dict) -> str:
    for key in ("question_type", "category", "cat", "type"):
        if key in row:
            return str(row[key])
    return "?"


def report(path: str, evid: dict[str, list[str]] | None = None) -> dict:
    evid = evid if evid is not None else {}
    rows = load_rows(path)
    by_cat: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    for row in rows:
        ctx = row.get("retrieved_context") or row.get("context") or ""
        if isinstance(ctx, list):
            ctx = " ".join(str(c) for c in ctx)
        # 优先用标注证据（LoCoMo），否则 gold 词，再退 rubric 要点（CLBench 无 gold）
        source = evid.get(str(row.get("id")))
        target = " ".join(source) if source else str(row.get("gold_answer") or "")
        if not target.strip():
            rub = row.get("rubric")
            if isinstance(rub, list):
                target = " ".join(str(r) for r in rub)
            elif rub:
                target = str(rub)
        rec, n_words = evidence_recall(target, ctx)
        if n_words == 0:
            continue
        by_cat[cat_of(row)].append(rec)
        all_scores.append(rec)
    name = path.replace("eval_input_", "").replace(".jsonl", "")
    if not all_scores:
        print(f"\n== {name}  (n=0, gold 实词全空, 跳过)")
        return {"name": name, "overall": 0.0, "by_cat": {}}
    print(f"\n== {name}  (n={len(all_scores)})  EvRecall={sum(all_scores)/len(all_scores):.3f}")
    for cat, scores in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        print(f"   {cat[:38]:40} n={len(scores):4}  {sum(scores)/len(scores):.3f}")
    return {"name": name, "overall": sum(all_scores) / len(all_scores), "by_cat": {c: sum(s)/len(s) for c, s in by_cat.items()}}


BASELINES = [
    "outputs/eval_input_session_bm25.jsonl",        # LoCoMo v0.1
    "outputs/eval_input_lme_s_message.jsonl",       # LongMemEval-S
    "outputs/eval_input_beam_100k_message.jsonl",   # BEAM-100K
    "outputs/eval_input_pmem2_message_top500.jsonl",# PersonaMem
    "outputs/eval_input_clbench_chunk.jsonl",       # CLBench
]

# LoCoMo evidence 标注（真 EvRecall：标注证据消息的实词在 context 的覆盖率，与 InvMem 0.9276 同口径）
LOCOMO_QUESTIONS = os.path.join(ROOT, "..", "LoCoMo_refined", "data", "public", "questions.jsonl")


def locomo_evidence() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    try:
        with open(LOCOMO_QUESTIONS, encoding="utf-8") as h:
            for line in h:
                if not line.strip():
                    continue
                q = json.loads(line)
                raw = q.get("evidence_messages", "[]")
                msgs = eval(raw) if isinstance(raw, str) else raw
                texts = [m.get("text", "") for m in msgs if m.get("text")]
                if texts:
                    mapping[str(q.get("qa_id"))] = texts
    except FileNotFoundError:
        pass
    return mapping


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        paths = args
    elif "--all" in sys.argv:
        paths = BASELINES
    else:
        paths = glob.glob(os.path.join(ROOT, "outputs", "eval_input_*.jsonl"))
    evid = locomo_evidence()
    def _ok(p: str) -> bool:
        if ".answers." in p or ".judged." in p:
            return False
        base = os.path.basename(p)
        return not (base.startswith("eval_input_") and "_top20" in base or "_top30" in base or ("_top10" in base))
    results = [report(p, evid) for p in paths if _ok(p)]
    if len(results) > 1:
        print(f"\n== 平均 EvRecall: {sum(r['overall'] for r in results)/len(results):.3f}  (LoCoMo 用标注证据, 其余用 gold 词近似)")


if __name__ == "__main__":
    main()
