#!/bin/bash
# ali_fixup.sh — 编排完成后的 error 补漏 pass：清 error 行 → 重跑 answers（done 跳过）→ 重判
cd /home/hezhi/AML/aml-system || exit 1
set -a; source .env; set +a
export PIPELINE_ALI_MODE=1 PIPELINE_JSON_MODE=0 PIPELINE_CONCURRENCY=4 PYTHONIOENCODING=utf-8

declare -A INPUTS=(
  [locomo_v03]="outputs/experiments/v03/eval_input_v2chunk_bm25.jsonl|local"
  [locomo_ablation]="outputs/experiments/ablation_time/eval_input_v2chunk_bm25.jsonl|local"
  [clb_v03]="outputs/experiments/v03/eval_input_clbench_chunk.jsonl|beam"
  [beam_e2a]="outputs/experiments/e2a/eval_input_beam_100k_message.jsonl|beam"
  [beam_e2b]="outputs/experiments/e2b/eval_input_beam_100k_message.jsonl|beam"
  [locomo_base]="outputs/eval_input_session_bm25.jsonl|local"
  [lme_base]="outputs/eval_input_lme_s_message.jsonl|local"
  [pmem_base]="outputs/eval_input_pmem2_message_top500.jsonl|local"
  [clb_base]="outputs/eval_input_clbench_chunk.jsonl|beam"
  [beam_base]="outputs/eval_input_beam_100k_message.jsonl|beam"
  [lme_v03]="outputs/experiments/lme_msg/eval_input_lme_s_message.jsonl|local"
  [pmem_v03]="outputs/experiments/epmem/eval_input_pmem2_message_top500.jsonl|local"
)

for name in "${!INPUTS[@]}"; do
  IFS='|' read -r inp proto <<< "${INPUTS[$name]}"
  ans="outputs/ali/${name}.answers.jsonl"
  [ -f "$ans" ] || continue
  n_err=$(grep -c '"error"' "$ans" 2>/dev/null || echo 0)
  if [ "$n_err" = "0" ]; then
    echo "[fixup] $name 无 error，跳过"
    continue
  fi
  echo "[fixup] $name 补 $n_err 题"
  venv/bin/python - "$ans" << 'PYEOF'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p, encoding="utf-8")]
ok = [r for r in rows if not r.get("error") and r.get("generated_answer", "").strip()]
open(p, "w", encoding="utf-8").writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in ok)
print(f"  清理后 {len(ok)} 行")
PYEOF
  if [ "$proto" = "local" ]; then
    venv/bin/python scripts/pipeline_local.py answer --input "$inp" --output "$ans"
    venv/bin/python scripts/pipeline_local.py evaluate --input "$inp" --answers "$ans" --output "outputs/ali/${name}.judged.jsonl"
  else
    venv/bin/python scripts/pipeline_beam.py answer --input "$inp" --output "$ans"
    venv/bin/python scripts/pipeline_beam.py evaluate --input "$inp" --answers "$ans" --output "outputs/ali/${name}.judged.jsonl"
  fi
  echo "[fixup] $name DONE"
done
echo "[fixup] 全部补漏完成 $(date +%H:%M)"
