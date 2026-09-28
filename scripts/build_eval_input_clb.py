# -*- coding: utf-8 -*-
"""CL-bench-Life 本地评测 harness（rubric 制，复用 pipeline_beam 评测）。

数据: E:/memoryboard/reference/CLBench-data/clbench_life.jsonl (405 样本)
结构: {messages: [user(超长context), assistant, ..., user(问题)], rubrics: [...], metadata: {task_id, ...}}

映射:
- user_id = clb_{task_id 前8位}; 末条 user 消息 = 问题, 其余全部切块入库
- 首条超长消息(可达 183K 字符)按平台规则 2000 词切块 Add
- 无 timestamp -> 合成递增
- 评测行带 rubric 字段 -> pipeline_beam answer+evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.store_v2 import MessageMemoryStore

DATA = Path("E:/memoryboard/reference/CLBench-data/clbench_life.jsonl")
WORD_SPLIT = 2000


def sanitize(text: str) -> str:
    for ch in (" ", " ", "", "", "", "", "", ""):
        text = text.replace(ch, " ")
    return text


def split_long(content: str, word_limit: int = WORD_SPLIT) -> list[str]:
    """超长单条消息按词数切成多段(近似平台 Add 的 2000 词分段规则)。"""
    words = content.split()
    if len(words) <= word_limit:
        return [content]
    return [" ".join(words[i: i + word_limit]) for i in range(0, len(words), word_limit)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").split("\n") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    store = MessageMemoryStore(granularity="chunk")
    questions = []
    for r in rows:
        tid = r["metadata"]["task_id"][:8]
        user_id = f"clb_{tid}"
        msgs = r["messages"]
        qmsg = msgs[-1]
        if qmsg["role"] != "user":  # 兜底: 找最后一条 user
            qmsg = next((m for m in reversed(msgs) if m["role"] == "user"), None)
        hist = msgs[: msgs.index(qmsg)] if qmsg in msgs else msgs[:-1]
        for hi, m in enumerate(hist):
            for piece in split_long(sanitize(m["content"])):
                store.add(user_id, f"{user_id}:h{hi}", [{
                    "role": m["role"], "content": piece, "timestamp": None,
                }])
        if qmsg is not None:
            questions.append((user_id, sanitize(qmsg["content"]), r))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "clbench_chunk" + (f"_top{args.limit}" if args.limit else "")
    eval_path = out / f"eval_input_{tag}.jsonl"
    n_chars = 0
    with eval_path.open("w", encoding="utf-8") as fh:
        for idx, (user_id, qtext, r) in enumerate(questions):
            hits = store.search(user_id, qtext, top_k=100)
            context = "\n\n".join(f"[memory {h['id']}] {h['content']}" for h in hits)
            n_chars += len(context)
            fh.write(json.dumps({
                "id": f"clb#{idx}",
                "question": qtext,
                "retrieved_context": context,
                "rubric": r["rubrics"],
                "question_type": r["metadata"].get("context_subcategory", "?"),
                "gold_answer": "",
                "category": r["metadata"].get("context_category", "?"),
            }, ensure_ascii=False) + "\n")
    print(f"[add/search] {len(questions)} questions -> {eval_path}")
    print(f"[stats] avg context {n_chars // max(1, len(questions))} chars")


if __name__ == "__main__":
    main()
