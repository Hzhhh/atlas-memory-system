# -*- coding: utf-8 -*-
"""badcase 抽样器：按题型打印错误题样本（question/gold/answer/证据命中）。

用法:
  python scripts/badcase_sample.py <eval_input.jsonl> <judged.jsonl> <题型关键词> [N=3]
"""
import json
import sys

sys.path.insert(0, ".")
from evrecall import evidence_recall  # noqa: E402


def main() -> None:
    inp, jud, kw = sys.argv[1], sys.argv[2], sys.argv[3]
    n_show = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    meta = {}
    with open(inp, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                meta[str(r["id"])] = r
    # answers 与 judged 同目录同名（.judged. -> .answers.），取模型原答
    answers: dict[str, str] = {}
    try:
        with open(jud.replace(".judged.", ".answers."), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    a = json.loads(line)
                    answers[str(a.get("id"))] = str(
                        a.get("generated_answer") or a.get("answer")
                        or a.get("model_answer") or a.get("text") or "")
    except FileNotFoundError:
        pass
    shown = 0
    with open(jud, encoding="utf-8") as f:
        for line in f:
            if shown >= n_show or not line.strip():
                continue
            j = json.loads(line)
            ok = j.get("is_correct")
            if j.get("llm_judge_score") is not None:
                ok = float(j["llm_judge_score"] or 0) >= 1
            if ok:
                continue
            m = meta.get(str(j.get("id")), {})
            cat = str(m.get("category") or m.get("question_type") or "?")
            if kw.lower() not in cat.lower():
                continue
            q = str(m.get("question", ""))[:180].replace("\n", " ")
            gold = str(m.get("gold_answer", ""))[:150].replace("\n", " ")
            ans = answers.get(str(j.get("id")), "")[:150].replace("\n", " ")
            ctx = m.get("retrieved_context") or m.get("context") or ""
            if isinstance(ctx, list):
                ctx = " ".join(str(c) for c in ctx)
            rec, nwords = evidence_recall(str(m.get("gold_answer", "")), str(ctx))
            print(f"--- [{cat}] id={j.get('id')} recall={rec:.2f}({nwords}词)")
            print(f"  Q: {q}")
            print(f"  GOLD: {gold}")
            print(f"  ANS: {ans or '(空/未记录)'}")
            shown += 1
    if shown == 0:
        print(f"[{kw}] 该题型未找到错误样本（或题型名不匹配）")


if __name__ == "__main__":
    main()
