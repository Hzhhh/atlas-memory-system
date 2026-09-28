# -*- coding: utf-8 -*-
"""按 user_id 隔离的记忆存储 + 检索。

检索模式（RETRIEVER 环境变量切换）:
- full : 返回该 user_id 全部记忆（暴力上限基线）
- bm25 : 词法检索
- embed: 向量检索（后续接入 OpenAI 兼容 embedding）
"""
from __future__ import annotations

import threading

from rank_bm25 import BM25Okapi

from .chunker import STRATEGIES


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class MemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # user_id -> {"memories": [ {id, content, ts} ], "bm25": BM25Okapi|None}
        self._users: dict[str, dict] = {}

    # ---------- Add ----------
    def add(self, user_id: str, session_id: str, messages: list[dict], strategy: str = "session") -> int:
        units = STRATEGIES[strategy](messages)
        with self._lock:
            bucket = self._users.setdefault(user_id, {"memories": [], "bm25": None, "next": 0})
            for unit in units:
                mem_id = f"{user_id}#{bucket['next']}"
                bucket["next"] += 1
                bucket["memories"].append({
                    "id": mem_id,
                    "content": unit["content"],
                    "ts": unit.get("ts"),
                    "session_id": session_id,
                })
            bucket["bm25"] = None  # 标记需重建索引
        return len(units)

    # ---------- Search ----------
    def search(self, user_id: str, query: str, top_k: int = 100, mode: str = "bm25") -> list[dict]:
        with self._lock:
            bucket = self._users.get(user_id)
            if not bucket:
                return []
            memories = list(bucket["memories"])
            bm25 = bucket["bm25"]

        if mode == "full" or not query.strip():
            ranked = [(m, 0.0) for m in memories]
        else:
            if bm25 is None:
                bm25 = BM25Okapi([_tokenize(m["content"]) for m in memories])
                with self._lock:
                    if self._users.get(user_id, {}).get("bm25") is None:
                        self._users[user_id]["bm25"] = bm25
            scores = bm25.get_scores(_tokenize(query))
            ranked = sorted(zip(memories, scores), key=lambda x: x[1], reverse=True)

        results = []
        for m, s in ranked[:top_k]:
            results.append({
                "id": m["id"],
                "content": m["content"],
                "score": round(float(s), 4),
            })
        return results

    # ---------- 运维 ----------
    def stats(self) -> dict:
        with self._lock:
            return {uid: len(b["memories"]) for uid, b in self._users.items()}

    def clear(self, user_id: str | None = None) -> None:
        with self._lock:
            if user_id:
                self._users.pop(user_id, None)
            else:
                self._users.clear()
