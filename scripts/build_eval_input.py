# -*- coding: utf-8 -*-
"""本地评测 harness：模拟 AML 平台的 Add → Search 流程，产出官方 pipeline 输入。

用法:
  venv/Scripts/python.exe scripts/build_eval_input.py --limit 20
  venv/Scripts/python.exe scripts/build_eval_input.py --strategy session --retriever bm25

流程:
  1. 按"来源会话"分组消息，模拟平台 20条消息/2000词 切 chunk，解析 session 时间为 timestamp
  2. 调 MemoryStore.add / .search（与线上 /add /search 同一实现）
  3. 输出 eval_input.jsonl: {id, question, retrieved_context, speaker_1/2_name, gold_answer}
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

DATA = Path("E:/memoryboard/LoCoMo_refined/data/public")


def parse_session_time(s: str) -> int | None:
    """'1:56 pm on 8 May, 2023' -> Unix 毫秒"""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return int(datetime.strptime(s, fmt).timestamp() * 1000)
        except ValueError:
            continue
    return None


def to_platform_chunks(sessions: list[dict]) -> list[list[dict]]:
    """模拟平台侧切法：每 session 一次 Add，>20 条消息或 >2000 词再分段。"""
    chunks = []
    for sess in sessions:
        msgs = []
        for m in sess["messages"]:
            text = m.get("text", "")
            caption = m.get("blip_caption", "")
            content = f"{text} [image: {caption}]" if caption else text
            msgs.append({
                "role": m.get("role", "user"),
                "content": content,
                "timestamp": parse_session_time(m.get("session_date_time", "")),
            })
        cur, cur_words = [], 0
        for m in msgs:
            w = len(m["content"].split())
            if cur and (len(cur) >= 20 or cur_words + w > 2000):
                chunks.append(cur)
                cur, cur_words = [], 0
            cur.append(m)
            cur_words += w
        if cur:
            chunks.append(cur)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 题（0=全部）")
    ap.add_argument("--strategy", default="session", choices=["session", "message", "window", "llm_cards", "v2", "v2chunk"])
    ap.add_argument("--seed-top-n", type=int, default=20, help="v2: 邻居扩展的种子数")
    ap.add_argument("--neighbor-span", type=int, default=1, help="v2: 每个种子前后各拉几条相邻消息")
    ap.add_argument("--retriever", default="bm25", choices=["full", "bm25"])
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    convs = [json.loads(l) for l in (DATA / "conversations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = [json.loads(l) for l in (DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        qs = qs[: args.limit]

    store = MemoryStore()
    store_v2 = MessageMemoryStore(
        seed_top_n=args.seed_top_n, neighbor_span=args.neighbor_span,
        granularity="chunk" if args.strategy == "v2chunk" else "message",
    )

    # ---- Add 阶段 ----
    total_chunks = 0
    for conv in convs:
        user_id = conv["sample_id"]
        chunks = to_platform_chunks(conv["sessions"])
        for i, chunk in enumerate(chunks):
            if args.strategy in ("v2", "v2chunk"):
                n = store_v2.add(user_id, f"{user_id}:session-{i}", chunk)
            else:
                n = store.add(user_id, f"{user_id}:sess-chunk-{i}", chunk, strategy=args.strategy)
            total_chunks += n
    print(f"[add] {len(convs)} conversations -> {total_chunks} memory units (strategy={args.strategy})")

    # ---- Search 阶段 ----
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_path = out / f"eval_input_{args.strategy}_{args.retriever}{'_top' + str(args.limit) if args.limit else ''}.jsonl"
    n_chars = 0
    with eval_path.open("w", encoding="utf-8") as fh:
        for q in qs:
            user_id = q["sample_id"]
            if args.strategy in ("v2", "v2chunk"):
                hits = store_v2.search(user_id, q["question"], top_k=100)
            else:
                hits = store.search(user_id, q["question"], top_k=100, mode=args.retriever)
            context = "\n\n".join(f"[memory {h['id']}] {h['content']}" for h in hits)
            n_chars += len(context)
            rec = {
                "id": q["qa_id"],
                "question": q["question"],
                "retrieved_context": context,
                "speaker_1_name": q.get("speaker_a", "speaker 1"),
                "speaker_2_name": q.get("speaker_b", "speaker 2"),
                "gold_answer": q["answer"],
                "category": q["category"],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[search] {len(qs)} questions -> {eval_path}")
    print(f"[stats] 平均每题检索上下文 {n_chars // max(1, len(qs))} 字符")


if __name__ == "__main__":
    main()
