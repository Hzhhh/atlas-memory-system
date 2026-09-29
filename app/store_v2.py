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
        self._bm25_build_lock = threading.Lock()  # 并发 search 去重（只 1 个线程构建，其余等缓存）
        self._dense_locks: dict[str, threading.Lock] = {}  # per-user dense 编码锁
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
    def add(self, user_id: str, session_id: str, messages: list[dict],
            request_id: str = "") -> int:
        with self._lock:
            bucket = self._users.setdefault(
                user_id, {"mems": [], "bm25": None, "dense": None, "seq": 0,
                          "uc": UpdateConflictIndex(), "ei": EntityIndex(), "persona": PersonaCard(),
                          "_req_ids": set()}
            )
            # request_id 幂等（官方 Full 断点续跑硬性要求：重放 Add 不得重复写入）
            if request_id and request_id in bucket["_req_ids"]:
                return 0
            if request_id:
                bucket["_req_ids"].add(request_id)
            if self.persona_enabled:
                body = "\n".join(str(m.get("content", "")) for m in messages)
                bucket["persona"].add_session(extract_persona(body))
            if self.granularity == "chunk":
                self._add_chunk(bucket, session_id, messages)
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
                # tokenize 缓存：Add 算一次，BM25 重建复用（重建从 tokenize 主导变统计主导）
                entry["_tokens"] = _tokenize(entry["index_text"])
                bucket["mems"].append(entry)
                bucket["seq"] += 1
            # BM25 不置脏：水位（bm25_upto）+ delta 打分，不再反复全量重建
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
        # 矛盾对构建改惰性（Search 时 ensure_pairs）——Add 路径零成本
        # （Full ADD_RUNTIME_ERROR 教训：每次 Add 全量配对超线性变慢）

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
    def _bm25_scores(self, user_id: str, bm25: BM25Okapi, upto: int,
                     mems: list[dict], n: int, q: str) -> np.ndarray:
        """冻结索引打分 + 增量条目（水位后新增）线性补打分，零全量重建。

        官方 Add/Search 交错场景的 P99 灾难根修：原实现每次 Add 置脏、
        Search 全量重建（大库秒级×并发排队=P99 400s+，实测 2/秒）。
        delta 打分复用冻结 IDF 的 BM25 公式（排名近似足够 RRF 用）；
        delta 超阈值才整体重建（周期性，每次 1-2s 可接受）。
        upto 与 bm25 由调用方锁内同刻拿（配套，防并发错位）；n=一致性水位。"""
        scores = bm25.get_scores(_tokenize(q))
        if upto >= n:
            # 并发下缓存索引可能比本次快照新（别线程快照更大）：截断到本快照
            # 前 n 条 = mems[:n] 的 BM25 分，数学正确
            return scores[:n] if scores.shape[0] > n else scores
        delta = mems[upto:n]
        if delta:
            qtoks = _tokenize(q)
            idf = bm25.idf
            default_idf = float(np.mean(list(idf.values()))) if idf else 1.0
            avgdl = max(bm25.avgdl, 1.0)
            extra = np.zeros(len(delta))
            for i, m in enumerate(delta):
                toks = m.get("_tokens") or _tokenize(m["index_text"])
                s = 0.0
                for t in qtoks:
                    f = toks.count(t)
                    if f:
                        s += idf.get(t, default_idf) * (f * 2.2) / (
                            f + 1.2 * (0.25 + 0.75 * len(toks) / avgdl))
                extra[i] = s
            scores = np.concatenate([scores, extra])
            # 周期性整体重建（delta 积累过大时，下次 search 生效）
            if len(delta) > max(3000, int(0.3 * max(bm25.corpus_size, 1))):
                with self._bm25_build_lock:
                    new = BM25Okapi([m.get("_tokens") or _tokenize(m["index_text"]) for m in mems[:n]])
                    with self._lock:
                        if self._users.get(user_id, {}).get("bm25") is bm25:
                            self._users[user_id]["bm25"] = new
                            self._users[user_id]["bm25_upto"] = n
        return scores

    def _dense_matrix(self, user_id: str, mems: list[dict], n: int) -> np.ndarray | None:
        """dense 增量索引：缓存 (matrix, n)，只编码新增条目后 vstack。

        官方评测 Add/Search 可能交错——全量重编码在大库（CLB 16 万条）CPU 上
        单次数十分钟会触发超时；bge 逐条编码独立确定，增量=全量数值一致。
        并发去重：同库并发 search 只有一个线程编码增量（per-user 编码锁），
        其余等锁后直接用新缓存——GIL 竞争与重复编码双消除（P99 400s+ 实测教训）。
        n=一致性水位（并发 add 安全）。新增编码失败返回 None（调用方降级
        BM25 单路；旧缓存保留，下次重试）。
        """
        if not self.dense_enabled:
            return None
        with self._lock:
            cached = self._users.get(user_id, {}).get("dense")
        if cached is not None and cached[1] == n:
            return cached[0]
        with self._dense_lock(user_id):
            with self._lock:  # 等锁期间别人可能已补编码
                cached = self._users.get(user_id, {}).get("dense")
            if cached is not None and cached[1] == n:
                return cached[0]
            if cached is not None and cached[1] > n:
                return cached[0][:n]  # 缓存比水位新：截断到本次快照
            matrix, done = (cached[0], cached[1]) if cached is not None else (None, 0)
            if n > done:
                new = encode_docs([m["content"] for m in mems[done:n]])
                if new is None:
                    return None
                matrix = new if matrix is None else np.vstack([matrix, new])
            if matrix is None or matrix.shape[0] != n:
                return None
            with self._lock:
                cur = self._users.get(user_id, {}).get("dense") or (None, 0)
                if self._users.get(user_id) is not None and cur[1] < n:
                    self._users[user_id]["dense"] = (matrix, n)
            return matrix

    def _dense_lock(self, user_id: str):
        """per-user 编码锁（全局字典，惰性创建）。"""
        with self._lock:
            lk = self._dense_locks.get(user_id)
            if lk is None:
                lk = threading.Lock()
                self._dense_locks[user_id] = lk
            return lk

    def search(self, user_id: str, query: str, top_k: int = 100,
               options: list[str] | None = None) -> list[dict]:
        with self._lock:
            bucket = self._users.get(user_id)
            # 引用 + 一致性水位：锁内取 n=len(mems)，全程只用前 n 条
            # （纯引用会在并发 add 下产生长度竞态 IndexError——实测教训）
            mems = bucket["mems"] if bucket else []
            n = len(mems)
            bm25 = bucket.get("bm25") if bucket else None
            bm25_upto = bucket.get("bm25_upto", 0) if bucket else 0  # 与 bm25 配套同刻拿

        if not mems:
            return []

        # A3: 选择题 options 拼入检索 query（选项文本携带答案实体词）
        q = query
        if options:
            q = query + "\nOptions:\n" + "\n".join(options)

        if bm25 is None:
            with self._bm25_build_lock:
                with self._lock:  # double-check：等锁期间别人可能已建好
                    u0 = self._users.get(user_id)
                    bm25 = u0.get("bm25") if u0 else None
                    if bm25 is not None:
                        bm25_upto = u0.get("bm25_upto", 0)
                if bm25 is None:
                    # 首建：冻结全量索引 + 记录水位（后续新增走 delta 打分不重建）
                    bm25 = BM25Okapi([m.get("_tokens") or _tokenize(m["index_text"]) for m in mems[:n]])
                    with self._lock:
                        u = self._users.get(user_id)
                        if u is not None and u.get("bm25") is None:
                            u["bm25"] = bm25
                            u["bm25_upto"] = n
                    bm25_upto = n
        scores = self._bm25_scores(user_id, bm25, bm25_upto, mems, n, q)
        # 加权 RRF 融合（InvMem 冻结参数）：fused = 1.0/(k+rd) + 0.5/(k+rb)
        matrix = self._dense_matrix(user_id, mems, n)
        qvec = encode_query(q) if matrix is not None else None
        if matrix is not None and qvec is not None:
            sims = matrix @ qvec                       # 余弦（已归一化）
            lex_mask = scores > 0                      # InvMem：仅 BM25>0 的进词法排名
            dense_order = np.argsort(-sims)            # dense 全量排名
            lex_order = np.argsort(-scores[lex_mask]) if lex_mask.any() else np.array([], dtype=int)
            lex_local = np.empty(n, dtype=np.int64)    # 尺寸锚定一致性水位 n
            lex_global = np.flatnonzero(lex_mask)
            lex_local[lex_global[lex_order]] = np.arange(len(lex_global))
            dense_rank = np.empty(n, dtype=np.int64)
            dense_rank[dense_order] = np.arange(n)
            fused = DENSE_WEIGHT / (RRF_K + dense_rank + 1)
            fused[lex_mask] += LEXICAL_WEIGHT / (RRF_K + lex_local[lex_mask] + 1)
            order = np.lexsort((-sims, -fused))        # 主键 fused，并列按 dense
            ranked = [(mems[i], float(fused[i])) for i in order]
        else:
            ranked = sorted(zip(mems, scores), key=lambda x: x[1], reverse=True)

        # A1+A4: 种子 + 同 session 相邻消息（证据常在"命中句的下一句回复"）
        by_seq = {m["seq"]: m for m in mems[:n]}  # 一致性水位
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
                            (m for m in mems[:n] if m["seq"] in want),
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

        # 契约硬约束：data 不超 top_k（注入卡插头部后总数可能超限，截尾部低分）
        return self._render(user_id, picked[:top_k], q)[:top_k]

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
