# -*- coding: utf-8 -*-
"""BEAM 本地评测 harness（100K/500K/1M 档，10M 结构不同暂缓）。

数据: E:/memoryboard/BEAM/chats/{100K,500K,1M}/{lib_id}/
  chat.json: list of batch {batch_number, turns, time_anchor}; turns = list of session(msg list)
  probing_questions/probing_questions.json: 10 能力 × 2 题, 每题带 rubric[]

映射:
- user_id = beam_{size}_{lib_id}; session = "{batch_number}_{session_idx}"
- msg content 清洗 " ->-> N,N" 索引标记; timestamp = time_anchor "March-15-2024"
- 评测行: {id, question, retrieved_context, rubric, question_type, gold_answer(仅参考)}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.store_v2 import MessageMemoryStore

ROOT = Path("E:/memoryboard/BEAM/chats")
ARROW = re.compile(r"\s*->->\s*[\d,]+\s*$")


def parse_anchor(s: str) -> int | None:
    try:
        return int(datetime.strptime(s, "%B-%d-%Y").timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def sanitize(text: str) -> str:
    for ch in (" ", " ", "", "", "", "", "", ""):
        text = text.replace(ch, " ")
    return ARROW.sub("", text)


def iter_adds(store: MessageMemoryStore, size: str, lib: Path) -> int:
    chat = json.loads((lib / "chat.json").read_text(encoding="utf-8"))
    user_id = f"beam_{size}_{lib.name}"
    n = 0
    for batch in chat:
        anchor = batch.get("time_anchor", "")
        ts = parse_anchor(anchor)
        for sess_idx, session in enumerate(batch.get("turns", [])):
            msgs = [{"role": m["role"], "content": sanitize(m["content"]),
                     "timestamp": ts if ts else parse_anchor(m.get("time_anchor", ""))}
                    for m in session]
            n += store.add(user_id, f"{user_id}:b{batch.get('batch_number', 0)}_s{sess_idx}", msgs)
    return n


GOLD_KEYS = ("ideal_response", "ideal_answer", "ideal_summary", "answer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100K", help="逗号分隔: 100K,500K,1M")
    ap.add_argument("--granularity", default="message", choices=["chunk", "message"])
    ap.add_argument("--limit-libs", type=int, default=0, help="每档只跑前 N 库(0=全部)")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    store = MessageMemoryStore(granularity=args.granularity)

    questions = []
    total_units = 0
    for size in sizes:
        libs = sorted((ROOT / size).iterdir())
        if args.limit_libs:
            libs = libs[: args.limit_libs]
        for lib in libs:
            if not (lib / "chat.json").exists():
                continue
            total_units += iter_adds(store, size, lib)
            pq = json.loads((lib / "probing_questions" / "probing_questions.json").read_text(encoding="utf-8"))
            for qtype, qlist in pq.items():
                for qi, q in enumerate(qlist if isinstance(qlist, list) else [qlist]):
                    gold = next((q[k] for k in GOLD_KEYS if q.get(k)), "")
                    questions.append({
                        "id": f"beam_{size}_{lib.name}#{qtype}_{qi}",
                        "user_id": f"beam_{size}_{lib.name}",
                        "question": q["question"],
                        "rubric": q.get("rubric", []),
                        "question_type": qtype,
                        "gold_answer": gold,
                    })
    print(f"[add] {len(sizes)} 档 -> {total_units} memory units ({args.granularity})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "beam_" + "-".join(s.replace("K", "k") for s in sizes) + f"_{args.granularity}"
    eval_path = out / f"eval_input_{tag}.jsonl"
    n_chars = 0
    with eval_path.open("w", encoding="utf-8") as fh:
        for q in questions:
            hits = store.search(q["user_id"], q["question"], top_k=100)
            context = "\n\n".join(f"[memory {h['id']}] {h['content']}" for h in hits)
            n_chars += len(context)
            fh.write(json.dumps({
                "id": q["id"],
                "question": q["question"],
                "retrieved_context": context,
                "rubric": q["rubric"],
                "question_type": q["question_type"],
                "gold_answer": q["gold_answer"],
                "category": q["question_type"],
            }, ensure_ascii=False) + "\n")
    print(f"[search] {len(questions)} questions -> {eval_path}")
    print(f"[stats] avg context {n_chars // max(1, len(questions))} chars")


if __name__ == "__main__":
    main()
