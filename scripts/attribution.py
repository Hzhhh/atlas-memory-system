# -*- coding: utf-8 -*-
"""归因工具：证据在场 × 答案对错 四象限，按题型分布（用户决策的"证据在场率归因法"）。

用法:
  python scripts/attribution.py            # 五集主基线
  python scripts/attribution.py outputs/eval_input_beam_100k_message.jsonl

象限定义（PRESENT_THRESHOLD=0.6, 判分 binary is_correct / rubric llm_judge_score>=1）:
  在场+对   → 通过
  不在场+错 → 架构问题(检索层) —— 本计划主攻
  在场+错   → 呈现可干预(LATEST/CONFLICT/时间戳) 或 模型问题(挂起校准)
  不在场+对 → 蒙对/宽松判分
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from evrecall import BASELINES, evidence_recall, load_rows, cat_of

PRESENT_TH = 0.6


def judged_path(input_path: str) -> str:
    return input_path.replace(".jsonl", ".judged.jsonl")


def load_judged(path: str) -> dict[str, dict]:
    out = {}
    try:
        for row in load_rows(path):
            out[str(row.get("id"))] = row
    except FileNotFoundError:
        print(f"[warn] 无 judged 文件: {path}")
    return out


def score_of(j: dict | None) -> float | None:
    if j is None:
        return None
    if "llm_judge_score" in j:
        return float(j["llm_judge_score"] or 0)
    if "is_correct" in j:
        return 1.0 if j["is_correct"] else 0.0
    return None


def report(input_path: str, judged_override: str | None = None) -> None:
    rows = load_rows(input_path)
    judged = load_judged(judged_override or judged_path(input_path))
    quads: dict[str, dict[str, int]] = defaultdict(lambda: {"PP": 0, "PW": 0, "AW": 0, "AP": 0})
    for row in rows:
        j = judged.get(str(row.get("id")))
        s = score_of(j)
        if s is None:
            continue
        ctx = row.get("retrieved_context") or row.get("context") or ""
        if isinstance(ctx, list):
            ctx = " ".join(str(c) for c in ctx)
        rec, n = evidence_recall(str(row.get("gold_answer", "")), ctx)
        if n == 0:
            continue
        present = rec >= PRESENT_TH
        correct = s >= 0.999
        key = ("P" if present else "A") + ("P" if correct else "W")  # P/A=在场/缺席, P/W=对/错
        quads[cat_of(row)][key] += 1
    name = input_path.replace("eval_input_", "").replace(".jsonl", "")
    print(f"\n== {name}   (PP=在场对 PW=在场错 AW=缺席错 AP=缺席对)")
    print(f"   {'题型':38} {'PP':>4} {'PW':>4} {'AW':>4} {'AP':>4}   可挽回(AW占比)")
    tot = {"PP": 0, "PW": 0, "AW": 0, "AP": 0}
    for cat, q in sorted(quads.items(), key=lambda kv: -(kv[1]["AW"] + kv[1]["PW"])):
        n = sum(q.values())
        for k in tot:
            tot[k] += q[k]
        print(f"   {cat[:36]:38} {q['PP']:>4} {q['PW']:>4} {q['AW']:>4} {q['AP']:>4}   {q['AW']/n:>5.0%}")
    n = sum(tot.values())
    if n:
        print(f"   {'TOTAL':38} {tot['PP']:>4} {tot['PW']:>4} {tot['AW']:>4} {tot['AP']:>4}   检索可挽回 {tot['AW']/n:.0%} / 呈现或模型 {tot['PW']/n:.0%}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jud_map: dict[str, str] = {}
    for a in sys.argv[1:]:
        if a.startswith("--judged="):
            inp, jud = a[len("--judged="):].split("=", 1)
            jud_map[inp] = jud
    for p in (args or BASELINES):
        try:
            report(p, jud_map.get(p))
        except FileNotFoundError:
            print(f"[skip] {p}")


if __name__ == "__main__":
    main()
