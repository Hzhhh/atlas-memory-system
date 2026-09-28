# -*- coding: utf-8 -*-
"""记忆切分策略。

平台侧已按 "20 条消息 / 2000 词" 把会话切成 chunk 发给 /add；
我们内部的切分粒度是自由的设计空间。

策略:
- session:    平台 chunk 原样存储（≈335 词/chunk），content 前缀显式日期
- message:    逐条消息存储（对照组，验证"过碎"假设）
- window-N:   滑窗合并 N 条消息为一个记忆单元（折中方案）
"""
from __future__ import annotations

from datetime import datetime, timezone

WORD_SPLIT = 2000
MSG_SPLIT = 20


def _fmt_ts(ts_ms: int | None) -> str:
    if not ts_ms:
        return ""
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OverflowError, OSError):
        return ""


def chunk_as_is(messages: list[dict]) -> list[dict]:
    """平台 chunk 原样转记忆单元，content 带日期前缀。"""
    if not messages:
        return []
    dates = [_fmt_ts(m.get("timestamp")) for m in messages]
    date_tag = dates[0] if dates and dates[0] else ""
    lines = []
    for m, d in zip(messages, dates):
        role = m.get("role", "user")
        prefix = f"[{d}] " if d else ""
        lines.append(f"{prefix}{role}: {m.get('content', '')}")
    content = "\n".join(lines)
    if date_tag:
        content = f"[{date_tag}]\n" + content
    return [{"content": content, "ts": messages[0].get("timestamp")}]


def chunk_by_message(messages: list[dict]) -> list[dict]:
    """逐条消息存储（对照组）。"""
    units = []
    for m in messages:
        d = _fmt_ts(m.get("timestamp"))
        prefix = f"[{d}] " if d else ""
        units.append({
            "content": f"{prefix}{m.get('role', 'user')}: {m.get('content', '')}",
            "ts": m.get("timestamp"),
        })
    return units


def chunk_sliding_window(messages: list[dict], window: int = 10, overlap: int = 0) -> list[dict]:
    """滑窗合并：window 条消息为一个记忆单元。"""
    units = []
    step = max(1, window - overlap)
    for start in range(0, len(messages), step):
        piece = messages[start:start + window]
        sub = chunk_as_is(piece)
        if sub:
            units.extend(sub)
        if start + window >= len(messages):
            break
    return units


def chunk_llm_cards(messages: list[dict]) -> list[dict]:
    """原文 chunk + LLM 时间锚定事实卡 双路存储（实验 v0.2）。"""
    from . import enricher  # 延迟导入，避免无 LLM 依赖时影响其他策略

    units = chunk_as_is(messages)
    if not units:
        return units
    cards = enricher.enrich_chunk(units[0]["content"])
    for card in cards:
        units.append({"content": card, "ts": messages[0].get("timestamp"), "is_card": True})
    return units


STRATEGIES = {
    "session": chunk_as_is,
    "message": chunk_by_message,
    "window": chunk_sliding_window,
    "llm_cards": chunk_llm_cards,
}
