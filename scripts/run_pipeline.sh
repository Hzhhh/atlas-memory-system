#!/usr/bin/env bash
# 跑官方 AML pipeline：answer（gpt-4o-mini）+ evaluate（judge）
# 用法: ./scripts/run_pipeline.sh <eval_input.jsonl>
# 需要在 aml-system/.env 配置 key（见 .env.example）
set -euo pipefail

cd "$(dirname "$0")/.."
INPUT="${1:?用法: run_pipeline.sh <eval_input.jsonl>}"
STAMP="$(basename "$INPUT" .jsonl)"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

PIPELINE="scripts/pipeline_local.py"  # 官方逻辑, 仅修复 async-with 文件句柄 bug
ANSWERS="outputs/${STAMP}.answers.jsonl"
JUDGED="outputs/${STAMP}.judged.jsonl"

echo "[1/2] answer (model=${ANSWER_MODEL})"
python "$PIPELINE" answer --input "$INPUT" --output "$ANSWERS"

echo "[2/2] evaluate (judge=${JUDGE_MODEL})"
python "$PIPELINE" evaluate --input "$INPUT" --answers "$ANSWERS" --output "$JUDGED"

echo "[score] 按类别统计 accuracy:"
python - "$INPUT" "$JUDGED" <<'PY'
import json, sys
from collections import defaultdict
inputs = {r["id"]: r for r in map(json.loads, open(sys.argv[1], encoding="utf-8"))}
agg = defaultdict(lambda: [0, 0])
for line in open(sys.argv[2], encoding="utf-8"):
    r = json.loads(line)
    cat = inputs.get(r["id"], {}).get("category", "?")
    agg[cat][0] += int(r["is_correct"])
    agg[cat][1] += 1
    agg["TOTAL"][0] += int(r["is_correct"])
    agg["TOTAL"][1] += 1
for cat in sorted(agg):
    ok, n = agg[cat]
    print(f"  cat {cat}: {ok}/{n} = {ok/n*100:.1f}%")
PY
