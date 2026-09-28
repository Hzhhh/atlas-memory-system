# -*- coding: utf-8 -*-
"""LongMemEval-S 本地评测 harness：与 LoCoMo 流程同构（官方 pipeline 输入 schema 相同）。

数据: E:/memoryboard/LongMemEval/data/longmemeval_s_cleaned.json (500 题)
用法:
  venv/Scripts/python.exe scripts/build_eval_input_lme.py [--limit N] [--granularity chunk|message]

映射:
- 每题独立记忆库 user_id = question_id（LongMemEval 设定：每题独立 haystack）
- 每 session 一次 Add（模拟平台"按来源会话调用"）
- haystack_dates[i] "2023/05/20 (Sat) 02:21" -> timestamp
- gold = answer, category = question_type
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.store import MemoryStore
from app.store_v2 import MessageMemoryStore

DATA = Path("E:/memoryboard/LongMemEval/data/longmemeval_s_cleaned.json")


def parse_date(s: str) -> int | None:
    try:
        return int(datetime.strptime(s[:10], "%Y/%m/%d").timestamp() * 1000)
    except ValueError:
        return None


def sanitize(text: str) -> str:
    """清除会把 splitlines() 触发分裂的 Unicode 行边界字符。"""
    for ch in (" ", " ", "", "", "", "", "", ""):
        text = text.replace(ch, " ")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--granularity", default="chunk", choices=["chunk", "message"])
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    store = MessageMemoryStore(granularity=args.granularity)

    # ---- Add 阶段: 每题独立库，每 session 一次 Add ----
    total_units = 0
    for q in data:
        uid = q["question_id"]
        for sess, date in zip(q["haystack_sessions"], q["haystack_dates"]):
            msgs = [{"role": m["role"], "content": sanitize(m["content"]), "timestamp": parse_date(date)}
                    for m in sess]
            total_units += store.add(uid, f"{uid}:s", msgs)
    print(f"[add] {len(data)} questions -> {total_units} memory units ({args.granularity})")

    # ---- Search 阶段 ----
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"lme_s_{args.granularity}" + (f"_top{args.limit}" if args.limit else "")
    eval_path = out / f"eval_input_{tag}.jsonl"
    n_chars = 0
    with eval_path.open("w", encoding="utf-8") as fh:
        for q in data:
            uid = q["question_id"]
            hits = store.search(uid, q["question"], top_k=100)
            context = "\n\n".join(f"[memory {h['id']}] {h['content']}" for h in hits)
            n_chars += len(context)
            fh.write(json.dumps({
                "id": uid,
                "question": q["question"],
                "retrieved_context": context,
                "speaker_1_name": "user",
                "speaker_2_name": "assistant",
                "gold_answer": q["answer"],
                "category": q["question_type"],
            }, ensure_ascii=False) + "\n")
    print(f"[search] {len(data)} questions -> {eval_path}")
    print(f"[stats] avg context {n_chars // max(1, len(data))} chars")


if __name__ == "__main__":
    main()
