# -*- coding: utf-8 -*-
"""v0.2 记忆存储：逐条消息粒度 + 检索工程四件套（无 LLM）。

借鉴情报（4 个上榜项目源码分析）：
- InvMem(第1名): 邻居窗口扩展 + 时间戳注入 content + BM25
- ChronoHybridMem(第5): latent_message_text = "Speaker/Event date" 前缀进索引键
- aml-memory-mvp(第10): options 拼入 query、返回值带 ID/角色/时间戳行前缀

铁律：返回的是原文（带元数据前缀），不做任何 LLM 加工改写。
"""
from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timezone

import numpy as np
from rank_bm25 import BM25Okapi

from .addons.update_detector import (
    UpdateConflictIndex, conflict_record_content, latest_record_content,
    scan_entries, topic_words,
)
from .addons.entity_index import EntityIndex
from .addons.timeline import build_timeline_card, timeline_triggered
from .addons.persona import PersonaCard, extract_persona
from .addons.time_resolver import resolve_relative
from .dense import encode_docs, encode_query

_TOKEN = re.compile(r"[a-z0-9]+")

# 加权 RRF 融合参数（InvMem 第 1 名冻结配置：dense 1.0 / BM25 0.5 / k=60）
RRF_K = 60
DENSE_WEIGHT = 1.0
LEXICAL_WEIGHT = 0.5


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def fmt_date(ts_ms: int | None) -> str:
    """毫秒时间戳 -> '17 August 2023'（英文长日期，gpt-4o-mini 日期算术更稳）"""
    if not ts_ms:
        return "unknown date"
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%d %B %Y")
    except (ValueError, OverflowError, OSError):
        return "unknown date"


class MessageMemoryStore:
    """消息/chunk 双粒度存储。检索 = BM25(索引文本) + 邻居窗口扩展。

    granularity:
    - "message": 逐条消息（实验A v2，EvRecall 受检索上限约束，适合超大数据集）
    - "chunk":   平台 chunk 原样聚合（v2.1，LoCoMo 类小库 100% 覆盖 + 时序增强）
    """

    def __init__(self, seed_top_n: int = 20, neighbor_span: int = 1,
                 granularity: str = "message") -> None:
        self._lock = threading.Lock()
        self._users: dict[str, dict] = {}
        self.seed_top_n = seed_top_n        # 邻居扩展的种子数（InvMem 用 20）
        self.neighbor_span = neighbor_span  # 每个种子前后各拉几条相邻消息
        self.granularity = granularity
        self.dense_enabled = os.environ.get("AML_DENSE", "1") == "1"
        self.inject_enabled = os.environ.get("AML_INJECT", "1") == "1"
        self.time_enabled = os.environ.get("AML_TIME", "1") == "1"
        self.entity_enabled = os.environ.get("AML_ENTITY", "1") == "1"
        self.persona_enabled = os.environ.get("AML_PERSONA", "0") == "1"  # 默认关（LLM 成本）
        self.timeline_enabled = os.environ.get("AML_TIMELINE", "0") == "1"  # 时间线卡（实验）
        # 拒答友好截断：融合分低的长尾不返回（证据干净 → answer 敢说 Cannot infer）
        self.abstain_enabled = os.environ.get("AML_ABSTAIN", "0") == "1"
        self.abstain_floor = float(os.environ.get("AML_ABSTAIN_FLOOR", "0.004"))
        self.abstain_min_keep = int(os.environ.get("AML_ABSTAIN_MIN_KEEP", "10"))

    # ---------- Add ----------
    def add(self, user_id: str, session_id: str, messages: list[dict]) -> int:
        with self._lock:
            bucket = self._users.setdefault(
                user_id, {"mems": [], "bm25": None, "dense": None, "seq": 0,
                          "uc": UpdateConflictIndex(), "ei": EntityIndex(), "persona": PersonaCard()}
            )
            if self.persona_enabled:
                body = "\n".join(str(m.get("content", "")) for m in messages)
                bucket["persona"].add_session(extract_persona(body))
            if self.granularity == "chunk":
                self._add_chunk(bucket, session_id, messages)
                bucket["bm25"] = None
                self._scan_updates(bucket)
                return 1
            for m in messages:
                ts = m.get("timestamp")
                role = m.get("role", "user")
                entry = {
                    "seq": bucket["seq"],          # 全局插入序（同 session 内天然连续）
                    "session_id": session_id,
                    "role": role,
                    "ts": ts,
                    "content": m.get("content", ""),
                }
                # 索引键：说话人 + 事件日期 + 原文（只进 BM25 语料，不进返回值）
                entry["index_text"] = (
                    f"Speaker: {role}\nEvent date: {fmt_date(ts)}\n{entry['content']}"
                )
                bucket["mems"].append(entry)
                bucket["seq"] += 1
            bucket["bm25"] = None
            # dense 不置脏：_dense_matrix 按 (matrix, n) 条目数增量补编码
            self._scan_updates(bucket)
            return len(messages)

    @staticmethod
    def _scan_updates(bucket: dict) -> None:
        """Add 端规则提取：新增 entries 过更新对/矛盾对检测 + 实体倒排（零 LLM）。"""
        uc = bucket.get("uc")
        ei = bucket.get("ei")
        if uc is None:
            return
        fresh = [m for m in bucket["mems"] if m["seq"] > uc.scanned]
        if not fresh:
            return
        uc.scanned = fresh[-1]["seq"]
        if ei is not None:
            for m in fresh:
                ei.add(m["seq"], m["content"])
        updates, negs, asserts = scan_entries(fresh)
        uc.add_updates(updates)
        uc.add_neg_sides(negs)
        uc.add_assertives(asserts)
        uc.build_conflicts()

    def _add_chunk(self, bucket: dict, session_id: str, messages: list[dict]) -> None:
        """整 chunk 一个记忆单元：Event date 进索引键，返回时带日期前缀。"""
        ts = next((m.get("timestamp") for m in messages if m.get("timestamp")), None)
        roles = sorted({m.get("role", "user") for m in messages})
        body = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        entry = {
            "seq": bucket["seq"],
            "session_id": session_id,
            "role": "/".join(roles),
            "ts": ts,
            "content": body,
        }
        entry["index_text"] = (
            f"Speakers: {entry['role']}\nEvent date: {fmt_date(ts)}\n{body}"
        )
        bucket["mems"].append(entry)
        bucket["seq"] += 1

    # ---------- Search ----------
    def _dense_matrix(self, user_id: str, mems: list[dict]) -> np.ndarray | None:
        """dense 增量索引：缓存 (matrix, n)，只编码新增条目后 vstack。

        官方评测 Add/Search 可能交错——全量重编码在大库（CLB 16 万条）CPU 上
        单次数十分钟会触发超时；bge 逐条编码独立确定，增量=全量数值一致。
        新增编码失败返回 None（调用方降级 BM25 单路；旧缓存保留，下次重试）。
        """
        if not self.dense_enabled:
            return None
        with self._lock:
            cached = self._users.get(user_id, {}).get("dense")
        if cached is not None and cached[1] == len(mems):
            return cached[0]
        matrix, n = (cached[0], cached[1]) if cached is not None else (None, 0)
        if len(mems) > n:
            new = encode_docs([m["content"] for m in mems[n:]])
            if new is None:
                return None
            matrix = new if matrix is None else np.vstack([matrix, new])
        if matrix is None or matrix.shape[0] != len(mems):
            return None
        with self._lock:
            if self._users.get(user_id) is not None:
                self._users[user_id]["dense"] = (matrix, len(mems))
        return matrix

    def search(self, user_id: str, query: str, top_k: int = 100,
               options: list[str] | None = None) -> list[dict]:
        with self._lock:
            bucket = self._users.get(user_id)
            mems = list(bucket["mems"]) if bucket else []
            bm25 = bucket.get("bm25") if bucket else None

        if not mems:
            return []

        # A3: 选择题 options 拼入检索 query（选项文本携带答案实体词）
        q = query
        if options:
            q = query + "\nOptions:\n" + "\n".join(options)

        if bm25 is None:
            bm25 = BM25Okapi([_tokenize(m["index_text"]) for m in mems])
            with self._lock:
                if self._users.get(user_id, {}).get("bm25") is None:
                    self._users[user_id]["bm25"] = bm25
        scores = bm25.get_scores(_tokenize(q))

        # 加权 RRF 融合（InvMem 冻结参数）：fused = 1.0/(k+rd) + 0.5/(k+rb)
        matrix = self._dense_matrix(user_id, mems)
        qvec = encode_query(q) if matrix is not None else None
        if matrix is not None and qvec is not None:
            sims = matrix @ qvec                       # 余弦（已归一化）
            lex_mask = scores > 0                      # InvMem：仅 BM25>0 的进词法排名
            dense_order = np.argsort(-sims)            # dense 全量排名
            lex_order = np.argsort(-scores[lex_mask]) if lex_mask.any() else np.array([], dtype=int)
            lex_local = np.empty(len(mems), dtype=np.int64)
            lex_global = np.flatnonzero(lex_mask)
            lex_local[lex_global[lex_order]] = np.arange(len(lex_global))
            dense_rank = np.empty(len(mems), dtype=np.int64)
            dense_rank[dense_order] = np.arange(len(mems))
            fused = DENSE_WEIGHT / (RRF_K + dense_rank + 1)
            fused[lex_mask] += LEXICAL_WEIGHT / (RRF_K + lex_local[lex_mask] + 1)
            order = np.lexsort((-sims, -fused))        # 主键 fused，并列按 dense
            ranked = [(mems[i], float(fused[i])) for i in order]
        else:
            ranked = sorted(zip(mems, scores), key=lambda x: x[1], reverse=True)

        # A1+A4: 种子 + 同 session 相邻消息（证据常在"命中句的下一句回复"）
        by_seq = {m["seq"]: m for m in mems}
        picked: list[tuple[dict, float]] = []
        seen: set[int] = set()

        def take(m: dict, score: float = 0.0) -> None:
            if m["seq"] not in seen:
                seen.add(m["seq"])
                picked.append((m, score))

        for mem, score in ranked[: self.seed_top_n]:
            take(mem, score)
            s = mem["seq"]
            # 同 session 内按 seq 相邻的消息（前 neighbor_span 条 + 后 neighbor_span 条）
            for d in range(1, self.neighbor_span + 1):
                for nb in (by_seq.get(s - d), by_seq.get(s + d)):
                    if nb is not None and nb["session_id"] == mem["session_id"]:
                        take(nb)
            if len(picked) >= top_k:
                break
        # 种子+邻居不足 top_k 时按原排名补足
        if len(picked) < top_k:
            for mem, score in ranked:
                take(mem, score)
                if len(picked) >= top_k:
                    break

        # 阶段2 实体扩展：种子命中的实体 → 倒排带出全部相关条目（列表完整性）
        if self.entity_enabled and len(picked) < top_k:
            with self._lock:
                ei = self._users.get(user_id, {}).get("ei")
            if ei is not None:
                seed_text = " ".join(m["content"] for m, _ in picked[: self.seed_top_n])
                ents = ei.entities_in(seed_text)
                if ents:
                    want = set(ei.seqs_for(ents)) - seen
                    if want:
                        by_rank = {id(m): i for i, (m, _) in enumerate(ranked)}
                        extra = sorted(
                            (m for m in mems if m["seq"] in want),
                            key=lambda m: by_rank.get(id(m), 10 ** 9),
                        )
                        for m in extra:
                            if len(picked) >= top_k:
                                break
                            take(m, 0.0)

        # 拒答友好截断（仅 dense 融合模式，RRF 分数口径统一）：低置信长尾不硬凑 top_k
        if self.abstain_enabled and matrix is not None and len(picked) > self.abstain_min_keep:
            strong = [(m, s) for m, s in picked if s >= self.abstain_floor]
            if strong:
                picked = strong[:top_k] if len(strong) >= self.abstain_min_keep \
                    else picked[: self.abstain_min_keep]

        return self._render(user_id, picked[:top_k], q)

    def _render(self, user_id: str, picked: list[tuple[dict, float]], q: str) -> list[dict]:
        """返回侧组装：时间戳前缀 + 更新对/矛盾对条目注入（原文句+标注，非 LLM 改写）。"""
        items = [
            {
                # A2: 返回侧时间戳文本化（平台只消费 content，独立字段会被丢弃）
                # A5: 相对时间解析注解（yesterday→绝对日期，judge 禁相对↔绝对互换）
                "id": f"mem-{m['seq']}",
                "content": (
                    f"[mem-{m['seq']} | {m['role']} | {fmt_date(m['ts'])}] {m['content']}"
                    + (resolve_relative(m["content"], m["ts"]) if self.time_enabled else "")
                ),
                "score": round(float(score), 6),
                "ts": m["ts"],
            }
            for m, score in picked
        ]
        with self._lock:
            uc = self._users.get(user_id, {}).get("uc")
            persona = self._users.get(user_id, {}).get("persona")
        # 时间线卡：时序意图 → 检索 top-N 按 ts 重排成卡（组织证据，零生成）
        tl_hit = False
        if self.timeline_enabled:
            card = build_timeline_card(picked, q)
            if card:
                items.insert(0, {"id": "timeline-0", "content": card, "score": 9.3, "ts": None})
                tl_hit = True
        if uc is None or not self.inject_enabled:
            return items
        q_words = set(topic_words(q))
        conflict = uc.conflict_for(q_words)
        if conflict is not None:
            items.insert(0, {"id": "conflict-0", "content": conflict_record_content(conflict),
                             "score": 9.0, "ts": conflict.get("side_neg_ts")})
        # 互斥路由：时间线已命中则 LATEST 不触发（how many days 归时间线不归值卡）
        latest = None if (tl_hit or timeline_triggered(q)) else uc.latest_for(q_words, q)
        if latest is not None and conflict is None:
            items.insert(0, {"id": f"upd-{latest['seq']}", "content": latest_record_content(latest),
                             "score": 9.5, "ts": latest["ts"]})
        if persona is not None and self.persona_enabled:
            card = persona.render(query=q)
            if card:
                items.insert(0, {"id": "persona-0", "content": card, "score": 9.8, "ts": None})
        return items

    # ---------- 便捷构造 ----------
    @classmethod
    def chunk_store(cls) -> "MessageMemoryStore":
        return cls(granularity="chunk")

    # ---------- 运维 ----------
    def stats(self) -> dict:
        with self._lock:
            return {uid: len(b["mems"]) for uid, b in self._users.items()}

    def clear(self) -> None:
        with self._lock:
            self._users.clear()
