#!/bin/bash
# server_run.sh — 统一阿里云口径全量重跑（两段式：无torch组立即跑，torch组等依赖装完接续）
# 用法: cd /home/hezhi/AML/aml-system && nohup bash scripts/server_run.sh > logs/orchestrator.log 2>&1 &
cd /home/hezhi/AML/aml-system || exit 1
set -a; source .env; set +a
export PIPELINE_ALI_MODE=1 PIPELINE_JSON_MODE=0 PIPELINE_CONCURRENCY=5 PYTHONIOENCODING=utf-8
mkdir -p logs outputs/ali
ALI=outputs/ali
run_local() {
  local name=$1 inp=$2
  venv/bin/python scripts/pipeline_local.py answer --input "$inp" --output "$ALI/${name}.answers.jsonl"
  venv/bin/python scripts/pipeline_local.py evaluate --input "$inp" --answers "$ALI/${name}.answers.jsonl" --output "$ALI/${name}.judged.jsonl"
  echo "[$(date +%H:%M)] DONE $name" >> logs/progress.log
}
run_beam() {
  local name=$1 inp=$2
  venv/bin/python scripts/pipeline_beam.py answer --input "$inp" --output "$ALI/${name}.answers.jsonl"
  venv/bin/python scripts/pipeline_beam.py evaluate --input "$inp" --answers "$ALI/${name}.answers.jsonl" --output "$ALI/${name}.judged.jsonl"
  echo "[$(date +%H:%M)] DONE $name" >> logs/progress.log
}

echo "[$(date +%H:%M)] === 第一段：无torch组（build已就绪） ==="
run_local locomo_v03      outputs/experiments/v03/eval_input_v2chunk_bm25.jsonl
run_local locomo_ablation outputs/experiments/ablation_time/eval_input_v2chunk_bm25.jsonl
run_beam  clb_v03         outputs/experiments/v03/eval_input_clbench_chunk.jsonl
run_beam  beam_e2a        outputs/experiments/e2a/eval_input_beam_100k_message.jsonl
run_beam  beam_e2b        outputs/experiments/e2b/eval_input_beam_100k_message.jsonl
run_local locomo_base     outputs/eval_input_session_bm25.jsonl
run_local lme_base        outputs/eval_input_lme_s_message.jsonl
run_local pmem_base       outputs/eval_input_pmem2_message_top500.jsonl
run_beam  clb_base        outputs/eval_input_clbench_chunk.jsonl
run_beam  beam_base       outputs/eval_input_beam_100k_message.jsonl

echo "[$(date +%H:%M)] === 第二段：torch 组（build 用 bge_env：torch2.5.1+cu121 现成环境） ==="
BGE=/home/hezhi/bge_env/bin/python
export AML_DENSE=1 AML_INJECT=1 AML_TIME=1 AML_ENTITY=1
$BGE scripts/build_eval_input_lme.py --granularity message --out outputs/experiments/lme_msg
unset AML_DENSE AML_INJECT AML_TIME AML_ENTITY
run_local lme_v03         outputs/experiments/lme_msg/eval_input_lme_s_message.jsonl

export AML_DENSE=1 AML_PERSONA=1
export PERSONA_CACHE=/home/hezhi/AML/aml-system/outputs/experiments/epmem/persona_cache.jsonl
$BGE scripts/build_eval_input_pmem.py --limit 500 --out outputs/experiments/epmem
run_local pmem_v03        outputs/experiments/epmem/eval_input_pmem2_message_top500.jsonl
unset AML_DENSE AML_PERSONA PERSONA_CACHE

echo "[$(date +%H:%M)] === 全部完成 ===" >> logs/progress.log
