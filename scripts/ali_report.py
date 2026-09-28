# -*- coding: utf-8 -*-
"""阿里云口径全量结果汇总：13 组分数 + 五集 v0.3 vs 基线对照 + 综合分。

用法（服务器）: cd /home/hezhi/AML/aml-system && venv/bin/python scripts/ali_report.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALI = ROOT / "outputs" / "ali"

# 组名 -> (input 路径, 协议, 数据集, 版本)
GROUPS = {
    "locomo_v03":      ("outputs/experiments/v03/eval_input_v2chunk_bm25.jsonl", "binary", "LoCoMo", "v0.3"),
    "locomo_ablation": ("outputs/experiments/ablation_time/eval_input_v2chunk_bm25.jsonl", "binary", "LoCoMo", "消融TIME"),
    "clb_v03":         ("outputs/experiments/v03/eval_input_clbench_chunk.jsonl", "rubric", "CLBench", "v0.3"),
    "beam_e2a":        ("outputs/experiments/e2a/eval_input_beam_100k_message.jsonl", "rubric", "BEAM", "对照"),
    "beam_e2b":        ("outputs/experiments/e2b/eval_input_beam_100k_message.jsonl", "rubric", "BEAM", "注入"),
    "locomo_base":     ("outputs/eval_input_session_bm25.jsonl", "binary", "LoCoMo", "基线"),
    "lme_base":        ("outputs/eval_input_lme_s_message.jsonl", "binary", "LME", "基线"),
    "pmem_base":       ("outputs/eval_input_pmem2_message_top500.jsonl", "binary", "PersonaMem", "基线"),
    "clb_base":        ("outputs/eval_input_clbench_chunk.jsonl", "rubric", "CLBench", "基线"),
    "beam_base":       ("outputs/eval_input_beam_100k_message.jsonl", "rubric", "BEAM", "基线"),
    "lme_v03":         ("outputs/experiments/lme_msg/eval_input_lme_s_message.jsonl", "binary", "LME", "v0.3"),
    "pmem_v03":        ("outputs/experiments/epmem/eval_input_pmem2_message_top500.jsonl", "binary", "PersonaMem", "v0.3"),
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def score_group(name: str) -> dict | None:
    meta = GROUPS[name]
    judged_p = ALI / f"{name}.judged.jsonl"
    if not judged_p.exists():
        return None
    jud = load_rows(judged_p)
    inp_p = ROOT / meta[0]
    inp = {r["id"]: r for r in load_rows(inp_p)} if inp_p.exists() else {}
    by = defaultdict(lambda: [0, 0])
    tot = [0, 0]
    errs = 0
    for r in jud:
        if r.get("error"):
            errs += 1
        q = inp.get(r["id"], {})
        cat = str(q.get("category") or q.get("question_type") or "?")
        if meta[1] == "binary":
            ok = 1 if r.get("is_correct") else 0
        else:
            ok = float(r.get("llm_judge_score") or 0)
        by[cat][0] += ok; by[cat][1] += 1
        tot[0] += ok; tot[1] += 1
    return {"name": name, "dataset": meta[2], "version": meta[3],
            "total": tot[0] / tot[1] * 100 if tot[1] else 0, "n": tot[1],
            "errors": errs, "by_cat": {c: s / n * 100 for c, (s, n) in by.items()}}


def main() -> None:
    results = {}
    print(f"{'组':22} {'数据集':12} {'版本':8} {'分数':>6} {'n':>5} {'err':>4}")
    for name in GROUPS:
        r = score_group(name)
        if r is None:
            print(f"{name:22} — 未完成")
            continue
        results[name] = r
        print(f"{name:22} {r['dataset']:12} {r['version']:8} {r['total']:6.1f} {r['n']:5} {r['errors']:4}")
    # 五集对照（v0.3 vs 基线，阿里云口径内部对比）
    print("\n== 五集对照（阿里云口径） ==")
    pairs = [("LoCoMo", "locomo_base", "locomo_v03"),
             ("LongMemEval", "lme_base", "lme_v03"),
             ("BEAM", "beam_base", "beam_e2b"),
             ("PersonaMem", "pmem_base", "pmem_v03"),
             ("CLBench", "clb_base", "clb_v03")]
    ready, scores_b, scores_v = 0, [], []
    for ds, b, v in pairs:
        if b in results and v in results:
            ready += 1
            print(f"{ds:14} 基线 {results[b]['total']:5.1f} → v0.3 {results[v]['total']:5.1f}  Δ{results[v]['total']-results[b]['total']:+.1f}")
            scores_b.append(results[b]["total"]); scores_v.append(results[v]["total"])
    if ready == 5:
        print(f"{'综合(等权)':14} 基线 {sum(scores_b)/5:5.1f} → v0.3 {sum(scores_v)/5:5.1f}  Δ{sum(scores_v)/5-sum(scores_b)/5:+.1f}")
    else:
        print(f"(已就绪 {ready}/5 对)")


if __name__ == "__main__":
    main()
