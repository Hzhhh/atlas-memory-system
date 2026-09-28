# -*- coding: utf-8 -*-
"""Add 阶段的 LLM 记忆增强：把原始 chunk 重写为带绝对日期的事实卡。

设计要点（对准失分模式 M1 时序相对↔绝对转换失败）:
- 每条事实卡强制带绝对日期前缀，相对时间(yesterday/last year)在写入时就换算好
- 结果缓存到磁盘：同一 chunk 只增强一次，实验迭代时零额外成本
- 双语无关：输出英文（数据集为英文，Answer 模型英文语境）
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx

ENRICH_API_BASE = os.environ.get("ENRICH_API_BASE", "http://llm.demo.haizhi.com/v1")
ENRICH_API_KEY = os.environ.get("ENRICH_API_KEY", "dummy-not-needed")
ENRICH_MODEL = os.environ.get("ENRICH_MODEL", "qwen3-14b")
CACHE_PATH = Path(os.environ.get("ENRICH_CACHE", "outputs/enrich_cache.json"))

FACT_CARD_PROMPT = """You are a memory extraction engine. Convert the conversation excerpt below into a list of self-contained fact cards.

Rules:
1. One card per line. Each card MUST start with an absolute date in brackets like [2023-05-08] or [2023] (use the year only when the day is unknown).
2. Convert every relative time expression (yesterday, last week, last year, next month...) into an absolute date, anchored at the excerpt date shown in the excerpt.
3. Each card format: [date] Person: fact. Use full names. Include events, preferences, opinions, plans, relationships, and personal details.
4. Cards must be in English, self-contained (understandable without the excerpt), and preserve specific names/places/numbers.
5. Do not summarize away details; do not add facts not present in the excerpt.
6. Output ONLY the cards, one per line, no preamble.

Conversation excerpt:
"""


def _cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def enrich_chunk(content: str, client: httpx.Client | None = None) -> list[str]:
    """chunk(已带日期前缀) -> 事实卡列表。带磁盘缓存。"""
    key = hashlib.md5(content.encode("utf-8")).hexdigest()
    cache = _cache()
    if key in cache:
        return cache[key]

    own = client is None
    c = client or httpx.Client(timeout=300)
    try:
        for attempt in range(3):
            try:
                resp = c.post(
                    ENRICH_API_BASE.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {ENRICH_API_KEY}"},
                    json={
                        "model": ENRICH_MODEL,
                        "messages": [{"role": "user", "content": FACT_CARD_PROMPT + content}],
                        "temperature": 0,
                    },
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"].strip()
                break
            except (httpx.HTTPError, KeyError, ValueError):
                if attempt == 2:
                    return []  # 增强失败降级: 不产卡
        cards = [ln.strip(" -") for ln in text.splitlines() if ln.strip().startswith("[")]
        cache[key] = cards
        _save_cache(cache)
        return cards
    finally:
        if own:
            c.close()
