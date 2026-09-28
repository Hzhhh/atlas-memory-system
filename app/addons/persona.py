# -*- coding: utf-8 -*-
"""偏好画像卡（P1）：Add 端 LLM 兜底提取用户偏好/背景，检索时注入头部导航层。

实证根因（BASELINE_REPORT 根因2）：pmem 26.4%——问"kitchen project"gold 要求结合
"爱烤面包"偏好，BM25 词法召回不到（偏好藏语义细节）。dense 通道补召回 + persona 卡
恒注入头部双保险。

成本控制：
- 按 session 聚合调用（非每 chunk），md5 缓存（重跑零成本）
- 失败降级为空（Add 永不因 LLM 失败）
- AML_PERSONA=0 总开关（默认关——只在 pmem 实验组开启）
- 线上正式评测时同一套代码换 gpt-4o-mini 端点（api_config 一改）
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading

import httpx

PERSONA_API_BASE = os.environ.get("PERSONA_API_BASE", "http://llm.demo.haizhi.com/v1")
PERSONA_MODEL = os.environ.get("PERSONA_MODEL", "qwen3-14b")
PERSONA_API_KEY = os.environ.get("PERSONA_API_KEY", "dummy")
PERSONA_CACHE = os.environ.get("PERSONA_CACHE", "")  # 空则不落盘

PERSONA_PROMPT = """Extract the USER's stable preferences, likes, dislikes, personal background, habits and goals from this conversation segment.

Rules:
- One fact per line, format: [LIKES|DISLIKES|BACKGROUND|HABITS|GOALS] fact
- Only statements about the user (person speaking with role 'user'), not the assistant
- Only stable/durable traits, skip one-off chatter
- Prefer specifics (e.g. "enjoys baking fresh bread on weekends" not "likes cooking")
- Use absolute dates if mentioned; no relative wording
- Output only the fact lines, nothing else. If nothing stable, output nothing.

Conversation:
{content}"""

_lock = threading.Lock()
_cache: dict[str, list[str]] = {}
_written: set[str] = set()   # 已落盘的 key（追加式写入，进程被杀不丢历史）


def _load_cache() -> None:
    global _cache
    if _cache or not PERSONA_CACHE or not os.path.exists(PERSONA_CACHE):
        return
    try:
        with open(PERSONA_CACHE, encoding="utf-8") as h:
            for line in h:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    _cache[str(row.get("k"))] = row.get("v") or []
                    _written.add(str(row.get("k")))
                except json.JSONDecodeError:
                    continue  # 半写行容忍
    except OSError:
        pass


def _save_cache() -> None:
    """追加式：只写新增条目（一行一 json），进程被杀不丢已提取结果。"""
    if not PERSONA_CACHE:
        return
    try:
        with _lock:
            fresh = [(k, v) for k, v in _cache.items() if k not in _written]
            if not fresh:
                return
            with open(PERSONA_CACHE, "a", encoding="utf-8") as h:
                for k, v in fresh:
                    h.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
                    _written.add(k)
    except OSError:
        pass


def extract_persona(content: str, client: httpx.Client | None = None) -> list[str]:
    """session 内容 → 偏好事实行列表。缓存命中零成本；LLM 失败返回 []。"""
    key = hashlib.md5(content.encode("utf-8")).hexdigest()
    _load_cache()
    if key in _cache:
        return _cache[key]
    prompt = PERSONA_PROMPT.replace("{content}", content[:6000])
    own = client is None
    c = client or httpx.Client(timeout=120)
    payload = {"model": PERSONA_MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 400}
    if os.environ.get("PERSONA_ALI_MODE", "0") == "1":   # 阿里云非流式需显式关思考
        payload["enable_thinking"] = False
    try:
        for _ in range(3):
            try:
                r = c.post(
                    f"{PERSONA_API_BASE.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {PERSONA_API_KEY}",
                             "Content-Type": "application/json"},
                    json=payload,
                )
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"].strip()
                lines = [l.strip().lstrip("- ").strip()
                         for l in text.split("\n") if l.strip().startswith("[")]
                with _lock:
                    _cache[key] = lines
                _save_cache()
                return lines
            except Exception:
                continue
        return []
    finally:
        if own:
            c.close()


class PersonaCard:
    """user 级偏好聚合卡（Add 时累积，Search 时注入头部）。

    注入策略（负面触发）：pmem 建议类 query 大多不含偏好词面（正向触发查全
    仅 21-43% 会杀死收益），故默认注入；仅医疗/事实类 query 跳过（health
    -7.3 误伤史的根因：健康 gold 不需要偏好背景，注入反而干扰）。
    """

    # 负面词元：跳过注入的事实/医疗类 query 特征
    _SKIP_RE = re.compile(
        r"\b(pain|doctor|doctors|medic\w*|diagnos\w*|symptom\w*|condition|"
        r"treatment|therap\w+|appointment|clinic|hospital|insurance|claim|"
        r"address|social\s+security|ssn|password|invoice|billing|"
        r"what\s+did|when\s+did|where\s+did|who\s+did|how\s+many)\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._seen: set[str] = set()

    def add_session(self, lines: list[str]) -> None:
        for l in lines:
            if len(l) < 8:  # 过滤空类别行（"[HABITS] "）
                continue
            k = l.casefold()
            if k not in self._seen:
                self._seen.add(k)
                self.lines.append(l)

    def render(self, limit: int = 20, query: str = "") -> str:
        """卡片文本（注入头部导航层；空卡/负面触发返回 ''）。"""
        if not self.lines:
            return ""
        if query and self._SKIP_RE.search(query):
            return ""
        body = "\n".join(f"- {l}" for l in self.lines[:limit])
        return f"[USER PREFERENCE SUMMARY | stable traits extracted from memory history]\n{body}"
