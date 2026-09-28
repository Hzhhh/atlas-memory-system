# -*- coding: utf-8 -*-
"""五数据集 baseline 终报：题型×准确率全表 + 本地综合分 + 公榜能力维度对照。

用法: venv/Scripts/python.exe scripts/final_report.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

O = Path("outputs")

# (数据集名, eval_input基名, 评分制: binary|rubric)
DATASETS = [
    ("LoCoMo-Refined", "eval_input_session_bm25", "binary"),
    ("LongMemEval-S", "eval_input_lme_s_message", "binary"),
    ("BEAM-100K", "eval_input_beam_100k_message", "rubric"),
    ("PersonaMem-v2(500)", "eval_input_pmem2_message_top500", "binary"),
    ("CLBench-Life", "eval_input_clbench_chunk", "rubric"),
]

# 题型 -> 公榜能力维度映射
DIM_MAP = {
    # LoCoMo
    "1": "A 事实召回", "2": "C 时序", "3": "B 多跳推理", "4": "A/B 事实与开放域",
    # LongMemEval
    "single-session-user": "A 事实召回", "single-session-assistant": "A 事实召回",
    "single-session-preference": "E 个性化", "multi-session": "B 多跳推理",
    "temporal-reasoning": "C 时序", "knowledge-update": "D 记忆治理",
    # BEAM
    "information_extraction": "A 事实召回", "multi_session_reasoning": "B 多跳推理",
    "temporal_reasoning": "C 时序", "event_ordering": "C 时序",
    "knowledge_update": "D 记忆治理", "contradiction_resolution": "D 记忆治理",
    "summarization": "D 记忆治理", "abstention": "H 安全",
    "instruction_following": "G 规则执行", "preference_following": "E 个性化",
}

# 公榜学术榜同期各维度最高分(供对照)
LEADERBOARD_MAX = {
    "A 事实召回": 64.4, "B 多跳推理": 55.3, "C 时序": 41.4, "D 记忆治理": 65.0,
    "E 个性化": 67.4, "G 规则执行": 38.6, "H 安全": 90.4,
}


def load(base: str):
    inputs = {}
    p = O / f"{base}.jsonl"
    if not p.exists():
        return None, None
    for l in open(p, encoding="utf-8"):
        if l.strip():
            inputs[json.loads(l)["id"]] = json.loads(l)
    jp = O / f"{base}.judged.jsonl"
    if not jp.exists():
        return inputs, None
    judged = [json.loads(l) for l in open(jp, encoding="utf-8") if l.strip()]
    return inputs, judged


def main() -> None:
    dim_acc = defaultdict(lambda: [0.0, 0])
    overall = {}

    print("=" * 78)
    print("一、五数据集 题型×准确率 全表")
    print("=" * 78)
    for name, base, mode in DATASETS:
        inputs, judged = load(base)
        if not judged:
            print(f"\n▶ {name}: 未完成")
            continue
        by_cat = defaultdict(lambda: [0.0, 0])
        for r in judged:
            cat = str(inputs.get(r["id"], {}).get("category", "?"))
            score = 1.0 if r.get("is_correct") else (r.get("llm_judge_score") or 0.0)
            if "error" in r and r.get("llm_judge_score") is None and not r.get("is_correct"):
                score = 0.0
            by_cat[cat][0] += score
            by_cat[cat][1] += 1
            dim = DIM_MAP.get(cat)
            if dim:
                dim_acc[dim][0] += score
                dim_acc[dim][1] += 1
        total_score = sum(v[0] for v in by_cat.values())
        total_n = sum(v[1] for v in by_cat.values())
        overall[name] = total_score / total_n * 100 if total_n else 0
        print(f"\n▶ {name}: TOTAL {overall[name]:.1f}% ({int(total_n)}题, {'binary' if mode=='binary' else 'rubric'}制)")
        for cat in sorted(by_cat, key=lambda c: -by_cat[c][1]):
            s, n = by_cat[cat]
            dim = DIM_MAP.get(cat, "-")
            print(f"    {cat:<28} {s/n*100:5.1f}%  (n={n:<4}) -> {dim}")

    if overall:
        print("\n" + "=" * 78)
        print("二、本地综合分")
        print("=" * 78)
        for k, v in overall.items():
            print(f"  {k:<24} {v:.1f}")
        mean = sum(overall.values()) / len(overall)
        print(f"  {'LOCAL OVERALL':<24} {mean:.2f}  (equal-weight)")

    if dim_acc:
        print("\n" + "=" * 78)
        print("三、能力维度对照（我们 vs 公榜学术榜同期最高分）")
        print("=" * 78)
        print(f"  {'维度':<12} {'本地':>8} {'公榜max':>8}   覆盖题量")
        for dim in sorted(dim_acc, key=lambda d: -dim_acc[d][1]):
            s, n = dim_acc[dim]
            lb = LEADERBOARD_MAX.get(dim, float("nan"))
            print(f"  {dim:<12} {s/n*100:>7.1f}% {lb:>7.1f}%   n={n}")


if __name__ == "__main__":
    main()
