# -*- coding: utf-8 -*-
"""多数据集汇总评分器：各数据集 judged 结果 -> 类官方综合分（0-100）。

用法:
  venv/Scripts/python.exe scripts/aggregate_score.py \
      locomo=outputs/eval_input_session_bm25 \
      longmemeval_s=outputs/eval_input_lme_s_message
  （每个条目为 eval_input 基名，自动找 .judged.jsonl）

输出: 每数据集 TOTAL + 分类别 accuracy，最后综合分 = 各数据集 accuracy 均值 × 100
（官方 0-100 归一化的近似：官方按 benchmark 内归一后跨数据集聚合，此为本地近似口径）
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load(base: str) -> tuple[dict, dict]:
    inputs = {r["id"]: r for r in map(json.loads, open(f"{base}.jsonl", encoding="utf-8"))}
    judged = [json.loads(l) for l in open(f"{base}.judged.jsonl", encoding="utf-8")]
    return inputs, judged


def main() -> None:
    pairs = [a.split("=", 1) for a in sys.argv[1:]]
    if not pairs:
        sys.exit(__doc__)

    accs = {}
    print(f"{'dataset':<18} {'TOTAL':>8}   categories")
    print("-" * 70)
    for name, base in pairs:
        inputs, judged = load(base)
        by_cat = defaultdict(lambda: [0, 0])
        for r in judged:
            cat = str(inputs.get(r["id"], {}).get("category", "?"))
            by_cat[cat][0] += int(bool(r.get("is_correct")))
            by_cat[cat][1] += 1
            by_cat["__total"][0] += int(bool(r.get("is_correct")))
            by_cat["__total"][1] += 1
        ok, n = by_cat["__total"]
        acc = ok / n if n else 0.0
        accs[name] = acc
        cats = "  ".join(
            f"{c}:{by_cat[c][0]/by_cat[c][1]*100:.0f}%({by_cat[c][1]})"
            for c in sorted(k for k in by_cat if k != "__total")
        )
        print(f"{name:<18} {acc*100:>7.1f}%   {cats}")

    if accs:
        overall = sum(accs.values()) / len(accs) * 100
        print("-" * 70)
        print(f"LOCAL OVERALL (equal-weight mean x100): {overall:.2f}")


if __name__ == "__main__":
    main()
