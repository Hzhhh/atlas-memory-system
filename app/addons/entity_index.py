# -*- coding: utf-8 -*-
"""实体倒排索引 + Search 两阶段扩展（列表完整性）。

实证根因（BASELINE_REPORT 根因4）：gold 3 个宠物只答 2 个——同一实体的事实散落
几十个 session，单点检索召回不全。两阶段：融合种子命中实体 → 倒排带出全部相关条目。

纯规则零 LLM：首字母大写连续词组 + 功能词黑名单 + user 级频次≥2 过滤。
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

# 连续大写词组（Flask-Login / New York / Michael）；单字母 I 排除
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z'-]{1,}(?:\s+[A-Z][a-zA-Z'-]{1,})*\b")
# 句首/代词/功能词黑名单（大写出现但非实体）
_BLACKLIST = {
    "I", "My", "Me", "We", "Our", "You", "Your", "He", "His", "She", "Her", "They",
    "Them", "Their", "It", "Its", "What", "When", "Where", "Why", "How", "Who",
    "The", "A", "An", "And", "But", "Or", "So", "If", "Then", "This", "That",
    "These", "Those", "There", "Here", "Was", "Is", "Are", "Were", "Been", "Have",
    "Has", "Had", "Do", "Does", "Did", "Will", "Would", "Can", "Could", "Should",
    "Not", "No", "Yes", "Also", "Just", "Really", "Actually", "Well", "Okay", "Ok",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Today", "Tomorrow", "Yesterday",
}
_MIN_FREQ = 2          # user 级最低出现次数（过滤偶然大写）
_MAX_ENTITIES = 2000   # 索引容量保护


def extract_entities(text: str) -> list[str]:
    """抽候选实体（原文形式，未去重）。"""
    out = []
    for m in _ENTITY_RE.finditer(text):
        e = m.group(0).strip()
        # 词组内任一词在黑名单且整体是单词 → 排除；多词组保留（New York）
        if e in _BLACKLIST:
            continue
        if len(e) < 3:
            continue
        out.append(e)
    return out


class EntityIndex:
    """user 级实体倒排（Add 时增量维护，含频次过滤）。"""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._posts: dict[str, list[int]] = defaultdict(list)   # entity(lower) -> [seq]
        self._active: dict[str, list[int]] = {}                 # 频次达标后的冻结视图
        self._dirty = True

    def add(self, seq: int, content: str) -> None:
        seen: set[str] = set()
        for e in extract_entities(content):
            key = e.casefold()
            if key in seen:
                continue
            seen.add(key)
            self._counts[key] += 1
            self._posts[key].append(seq)
            self._dirty = True

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        self._active = {
            k: seqs for k, seqs in self._posts.items()
            if self._counts[k] >= _MIN_FREQ
        }
        self._dirty = False

    def entities_in(self, text: str) -> list[str]:
        """text 中出现且已入索引的实体（lower）。"""
        self._rebuild()
        hits = []
        for e in extract_entities(text):
            key = e.casefold()
            if key in self._active and key not in hits:
                hits.append(key)
        return hits

    def entities_in_all(self, text: str) -> list[str]:
        """低频可见版（频次≥1）：时序题实体常只出现一次（MoMA/Met），
        频次≥2 的 active 视图会漏。误伤由调用方的 query 词元意图控制。"""
        hits = []
        for e in extract_entities(text):
            key = e.casefold()
            if key in self._posts and key not in hits:
                hits.append(key)
        return hits

    def seqs_for_all(self, entity_keys: list[str], limit_per: int = 30) -> list[int]:
        """低频版倒排（时间线卡用：排序需要实体的全部条目，limit 放宽）。"""
        out: list[int] = []
        for k in entity_keys:
            out.extend(self._posts.get(k, [])[:limit_per])
        return out

    def seqs_for(self, entity_keys: list[str], limit_per: int = 8) -> list[int]:
        """实体 → 全部相关 seq（每实体截断保护）。"""
        self._rebuild()
        out: list[int] = []
        for k in entity_keys:
            out.extend(self._active.get(k, [])[:limit_per])
        return out

    def stats(self) -> dict:
        self._rebuild()
        return {"entities": len(self._active), "raw": len(self._posts)}
