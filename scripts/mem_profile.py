# -*- coding: utf-8 -*-
"""内存剖析 v2：CLB+BEAM 混合灌库，测单条目内存成本与结构分布，外推官方全量。

锚点: 新加坡 22831 add 时文本库约 9-10G（14G 满载 − 编码评测/常驻服务 4-5G）。
用法: python scripts/mem_profile.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("AML_DENSE", "0")
os.environ.setdefault("AML_PERSONA", "0")

from app.store_v2 import MessageMemoryStore

CLB = Path("E:/memoryboard/reference/CLBench-data/clbench_life.jsonl")
BEAM = Path("E:/memoryboard/BEAM/chats/100K")
ARROW = re.compile("[\u2190-\u21ff]")


def sanitize(text: str) -> str:
    # 测量用途：只去箭头符号，特殊空白字符对内存分布无影响（含量极低）
    return ARROW.sub("", text)


def split_long(text: str, limit: int = 6000) -> list[str]:
    """长消息切片（近似官方 2000 词分段）。"""
    return [text[i:i + limit] for i in range(0, len(text), limit)] or [""]


def main() -> None:
    store = MessageMemoryStore(granularity="message")
    tracemalloc.start()

    n_add = n_msg = n_char = 0
    # --- CLB: 超长 context 消息切片 ---
    rows = [json.loads(l) for l in CLB.read_text(encoding="utf-8").split("\n") if l.strip()]
    for r in rows:
        tid = r["metadata"]["task_id"][:8]
        uid = f"clb_{tid}"
        qmsg = r["messages"][-1]
        hist = r["messages"][:-1] if qmsg.get("role") == "user" else r["messages"]
        buf = []
        for hi, m in enumerate(hist):
            for piece in split_long(sanitize(m["content"])):
                buf.append({"role": m["role"], "content": piece, "timestamp": None})
        # 官方模式：每 20 条一个 add
        for i in range(0, len(buf), 20):
            chunk = buf[i:i + 20]
            n_add += store.add(uid, f"{uid}:h{i//20}", chunk, request_id=f"mp-c-{tid}-{i//20}") or 0
            n_msg += len(chunk)
            n_char += sum(len(c["content"]) for c in chunk)

    # --- BEAM: 短消息对话 ---
    from datetime import datetime
    def parse_anchor(s):
        try:
            return int(datetime.strptime(s, "%B-%d-%Y").timestamp() * 1000)
        except (ValueError, TypeError):
            return None

    for lib in sorted(p for p in BEAM.iterdir() if (p / "chat.json").exists()):
        chat = json.loads((lib / "chat.json").read_text(encoding="utf-8"))
        uid = f"beam_{lib.name}"
        for batch in chat:
            ts = parse_anchor(batch.get("time_anchor", ""))
            for si, session in enumerate(batch.get("turns", [])):
                msgs = [{"role": m["role"], "content": sanitize(m["content"]),
                         "timestamp": ts if ts else parse_anchor(m.get("time_anchor", ""))}
                        for m in session]
                n_add += store.add(uid, f"{uid}:b{batch.get('batch_number',0)}_s{si}", msgs, request_id=f"mp-b-{lib.name}-{si}") or 0
                n_msg += len(msgs)
                n_char += sum(len(m["content"]) for m in msgs)

    cur, peak = tracemalloc.get_traced_memory()
    print(f"灌库: {n_add} add / {n_msg} 条目 / {n_char/2**20:.0f}MB 文本 / {len(store._users)} users")
    print(f"Python 堆: 当前 {cur/2**20:.0f}MB / 峰值 {peak/2**20:.0f}MB")
    print(f"单条目成本: {cur/n_msg/1024:.1f}KB/条目")

    tot_a = tot_n = tot_c = 0
    for b in store._users.values():
        uc = b["uc"]
        tot_a += len(uc.assertives); tot_n += len(uc.neg_sides); tot_c += len(uc.chains)
    print(f"规则索引: assertives={tot_a} negs={tot_n} chains={tot_c}")

    snap = tracemalloc.take_snapshot()
    by_site = {}
    for stat in snap.statistics("lineno"):
        f = stat.traceback[0]
        by_site[f"{Path(f.filename).name}:{f.lineno}"] = by_site.get(f"{Path(f.filename).name}:{f.lineno}", 0) + stat.size
    print("\nTop12 分配点:")
    for k, v in sorted(by_site.items(), key=lambda x: -x[1])[:12]:
        print(f"  {v/2**20:7.1f}MB  {k}")

    # 触发 search（BM25 首建 + ensure_pairs）
    for uid in list(store._users)[:2]:
        store.search(uid, "what is the target deadline", top_k=100)
    cur2, peak2 = tracemalloc.get_traced_memory()
    print(f"\nsearch后 堆: {cur2/2**20:.0f}MB (BM25+pairs 增量 {(cur2-cur)/2**20:.0f}MB, 峰值 {peak2/2**20:.0f}MB)")


if __name__ == "__main__":
    main()
