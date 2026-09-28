# -*- coding: utf-8 -*-
"""相对时间解析：以消息时间戳为锚，把 yesterday/last year/N months ago 解析为绝对日期。

移植自 InvMem temporal_enrichment（AML 学术榜第 1 名，LoCoMo cat2 时序题验证有效）。
实测根因（BASELINE_REPORT 根因3）：gold"2022"答"last year"被判 WRONG——judge 禁相对↔绝对互换，
答案侧必须给绝对值，而绝对值只能从上下文里的解析注解推出来。
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_QUANTITIES = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_AGO_RE = re.compile(
    r"\b(?P<count>a|an|one|two|three|four|five|\d+)\s+(?P<unit>day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)


def _fmt(d: datetime) -> str:
    return f"{d:%d %B %Y}"


def _shift_months(d: datetime, months: int) -> datetime:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return d.replace(year=year, month=month, day=day)


def _fmt_week(d: datetime) -> str:
    return f"week of {_fmt(d - timedelta(days=d.weekday()))}"


def _fmt_month(d: datetime) -> str:
    return f"{d:%B %Y}"


def resolve_relative(content: str, ts_ms: int | None) -> str:
    """返回 '[Resolved relative dates: ...]' 注解文本；无相对词或无时间锚返回 ''。"""
    if not ts_ms:
        return ""
    try:
        anchor = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return ""
    rules = (
        (r"\blast night\b", _fmt(anchor - timedelta(days=1))),
        (r"\byesterday\b", _fmt(anchor - timedelta(days=1))),
        (r"\btomorrow\b", _fmt(anchor + timedelta(days=1))),
        (r"\b(?:today|tonight)\b", _fmt(anchor)),
        (r"\blast week\b", _fmt_week(anchor - timedelta(weeks=1))),
        (r"\bthis week\b", _fmt_week(anchor)),
        (r"\bnext week\b", _fmt_week(anchor + timedelta(weeks=1))),
        (r"\blast month\b", _fmt_month(_shift_months(anchor, -1))),
        (r"\bthis month\b", _fmt_month(anchor)),
        (r"\bnext month\b", _fmt_month(_shift_months(anchor, 1))),
        (r"\blast year\b", str(anchor.year - 1)),
        (r"\bthis year\b", str(anchor.year)),
        (r"\bnext year\b", str(anchor.year + 1)),
    )
    annotations: list[str] = []
    seen: set[str] = set()

    def push(phrase: str, resolved: str) -> None:
        key = phrase.casefold()
        if key not in seen:
            annotations.append(f"{phrase} = {resolved}")
            seen.add(key)

    for pattern, resolved in rules:
        for m in re.finditer(pattern, content, flags=re.IGNORECASE):
            push(m.group(0), resolved)

    for m in _AGO_RE.finditer(content):
        raw = m.group("count").casefold()
        count = _QUANTITIES.get(raw, int(raw) if raw.isdigit() else 1)
        unit = m.group("unit").casefold()
        if unit == "day":
            resolved = _fmt(anchor - timedelta(days=count))
        elif unit == "week":
            resolved = _fmt(anchor - timedelta(weeks=count))
        elif unit == "month":
            resolved = _fmt(_shift_months(anchor, -count))
        else:
            resolved = str(anchor.year - count)
        push(m.group(0), resolved)

    return f" [Resolved relative dates: {'; '.join(annotations)}]" if annotations else ""
