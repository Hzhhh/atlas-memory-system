# -*- coding: utf-8 -*-
"""PersonaMem-v2 文本版本地评测 harness。

数据:
- benchmark: E:/memoryboard/reference/PersonaMem-v2-data/benchmark_text.csv (5000 行)
- chat_history: E:/memoryboard/reference/PersonaMem-v2-data/chat_history_32k/personaN.json

映射:
- user_id = pmem2_{persona_id}; chat_history 无 session 边界 -> 按每 20 条消息合成一批 Add
- 无 timestamp -> 按序合成递增时间戳(基准日 + i 天)
- question = user_query['content'](ast.literal_eval 解析); gold = correct_answer
- 口径注记: 官方为 MCQ 准确率; 本地统一用开放 QA + binary judge(gold=correct_answer)
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.store_v2 import MessageMemoryStore

BASE = Path("E:/memoryboard/reference/PersonaMem-v2-data")
BATCH = 20
EPOCH = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
DAY = 86400_000


def sanitize(text: str) -> str:
    for ch in (" ", " ", "", "", "", "", "", ""):
        text = text.replace(ch, " ")
    return text


def load_history(persona_id: int) -> list[dict]:
    # 文件名时间戳部分固定, 直接按 persona{id} 匹配
    cands = list((BASE / "chat_history_32k").glob(f"*_persona{persona_id}.json"))
    if not cands:
        return []
    data = json.loads(cands[0].read_text(encoding="utf-8"))
    msgs = data.get("chat_history") or data.get("conversations") or []
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="评测前 N 题(0=全部5000)")
    ap.add_argument("--granularity", default="message", choices=["chunk", "message"])
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(BASE / "benchmark_text.csv", encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]

    store = MessageMemoryStore(granularity=args.granularity)
    questions = []
    hist_cache: dict[int, list[dict]] = {}
    total_units = 0
    for r in rows:
        pid = int(r["persona_id"])
        if pid not in hist_cache:
            hist_cache[pid] = load_history(pid)
        hist = hist_cache[pid]
        user_id = f"pmem2_{pid}"
        questions.append((user_id, r))

    added: set[str] = set()
    for user_id, r in questions:
        if user_id in added:
            continue
        pid = int(user_id.split("_")[1])
        hist = hist_cache.get(pid, [])
        for bi in range(0, len(hist), BATCH):
            batch = hist[bi: bi + BATCH]
            msgs = [{
                "role": m.get("role", "user"),
                "content": sanitize(str(m.get("content", ""))),
                "timestamp": EPOCH + (bi // BATCH) * DAY,
            } for m in batch]
            total_units += store.add(user_id, f"{user_id}:h{bi // BATCH}", msgs)
        added.add(user_id)
    print(f"[add] {len(added)} personas -> {total_units} memory units ({args.granularity})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"pmem2_{args.granularity}" + (f"_top{args.limit}" if args.limit else "")
    eval_path = out / f"eval_input_{tag}.jsonl"
    n_chars = 0
    with eval_path.open("w", encoding="utf-8") as fh:
        for idx, (user_id, r) in enumerate(questions):
            try:
                uq = ast.literal_eval(r["user_query"])
                qtext = uq.get("content", r["user_query"]) if isinstance(uq, dict) else str(uq)
            except (ValueError, SyntaxError):
                qtext = r["user_query"]
            hits = store.search(user_id, qtext, top_k=100)
            context = "\n\n".join(f"[memory {h['id']}] {h['content']}" for h in hits)
            n_chars += len(context)
            fh.write(json.dumps({
                "id": f"pmem2#{idx}",
                "question": qtext,
                "retrieved_context": context,
                "gold_answer": r.get("correct_answer", ""),
                "speaker_1_name": "user",
                "speaker_2_name": "assistant",
                "category": r.get("pref_type", "?"),
            }, ensure_ascii=False) + "\n")
    print(f"[search] {len(questions)} questions -> {eval_path}")
    print(f"[stats] avg context {n_chars // max(1, len(questions))} chars")


if __name__ == "__main__":
    main()
