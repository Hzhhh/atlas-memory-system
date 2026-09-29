# -*- coding: utf-8 -*-
"""Add 端规则提取层：更新对/矛盾对检测（零 LLM，零延迟）。

实证依据（EXPERIMENT_P0.md）：
- BEAM knowledge_update 新值在场率 82.5% 但 answer 选旧值 → 缺"哪个是最新"信号
- gold 模板 = "rescheduled to March 27" / "new target of 1,350 words"（marker 词触发）
- [LATEST VALUE RECORD] 条目实测翻转 6/18 低分题（qwen3-14b）
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

# 更新标记词（实证 badcase 语料归纳 + 常见扩展）
UPDATE_MARKERS = re.compile(
    r"\b(actually|now|updated?|changed?|reschedul\w*|no longer|previously|"
    r"instead of|decided to|switch(?:ed)?(?: to)?|moved to|revised|"
    r"new (?:target|goal|deadline|date|schedule|budget|number|total)|"
    r"bumped|increased to|decreased to|reduced to|raised to|lowered to|"
    r"cut (?:it|down) to|upped? (?:it|to)|set (?:it|the \w+) to|adjusted to|"
    r"final(?:ly)? (?:date|deadline|target)|as of (?:now|today))\b",
    re.IGNORECASE,
)
# 否定词（矛盾对检测：同主题肯定 vs 否定）
NEGATIONS = re.compile(r"\b(never|not|n't|no longer|haven'?t|hasn'?t|didn'?t|don'?t|doesn'?t|won'?t|can'?t)\b", re.IGNORECASE)

VALUE_RE = re.compile(
    r"(\$?\d[\d,.]*\s*(?:%|k|hours?|hrs?|words?|dollars?|bucks?|miles?|km|minutes?|mins?|days?|weeks?|months?|years?)?|"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b|"
    r"\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?\b)"
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]{3,}")

STOP = set("""actually now updated changed rescheduled longer previously instead decided
switched moved revised bumped increased decreased reduced raised lowered adjusted finally
will would have has had the a an and or but for with about from this that these those your
you i we they it its was were is are been said says just really very much more most some any
all can could should may might want wanted think thought know knew like love enjoy prefer
favorite thing things time today tomorrow yesterday one two three four five six seven eight nine ten
what which who whom whose when where why how
""".split())


def _stem(w: str) -> str:
    """轻量词干化：统一 weekly/week、words/word 词形（检索匹配用，不求语言学正确）。"""
    for suf, fix in (("ies", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + fix
    return w


def topic_words(sentence: str) -> list[str]:
    """句子的主题实词（去停词/marker 词，stem 化，保序去重）。"""
    out: list[str] = []
    for w in _WORD.findall(sentence):
        lw = w.lower()
        if lw in STOP:
            continue
        sw = _stem(lw)
        if sw not in out:
            out.append(sw)
    return out


def extract_values(sentence: str) -> list[str]:
    return [m.group(0).strip() for m in VALUE_RE.finditer(sentence)]


def _fmt_ts(ts) -> str:
    if not ts:
        return "unknown date"
    try:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%d %B %Y")
    except (ValueError, OverflowError, OSError):
        return "unknown date"


# 注入 v2：数值期望词（query 在"要一个值"时才允许 LATEST 注入，闲聊/描述性问句不触发）
_VALUE_EXPECT_RE = re.compile(
    r"\b(how\s+many|how\s+much|how\s+long|what\s+is|what\s+are|what\s+was|when\s+is|when\s+was|"
    r"when\s+does|when\s+did|which\s+\w+|do\s+i|am\s+i|have\s+i|is\s+my|are\s+my|was\s+my|"
    r"what(?:'s| is)\s+my|current|latest|target|budget|deadline|total)\b",
    re.IGNORECASE,
)


def _has_value_expectation(query: str) -> bool:
    """query 是否表达'期望得到一个具体值'（计数/数值/日期/状态确认类问法）。"""
    return bool(_VALUE_EXPECT_RE.search(query))


def scan_entries(entries: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """对新增 entries 扫描，返回 (更新链事件, 否定侧事件, 肯定侧事件)。

    entry 需含 seq/session_id/ts/content。事件结构:
    更新: {"topic": [w...], "value": str, "seq": int, "ts": ts, "sentence": str}
    否定/肯定: {"topic": [w...], "value": [v...], "seq": int, "ts": ts, "sentence": str}
    肯定侧（矛盾对正侧用）不要求 update marker——BEAM 实证：正侧句
    "Flask-Login v0.6.2 was integrated" 无 marker，marker 门槛使矛盾对
    触发率仅 8/40。
    """
    updates: list[dict] = []
    conflicts: list[dict] = []
    assertives: list[dict] = []
    for e in entries:
        body = e.get("content", "")
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                continue
            for sent in _SENT_SPLIT.split(line):
                sent = sent.strip()
                if len(sent) < 12:
                    continue
                tw = topic_words(sent)
                if len(tw) < 2:
                    continue
                vals = extract_values(sent)
                has_marker = bool(UPDATE_MARKERS.search(sent))
                has_neg = bool(NEGATIONS.search(sent))
                if has_marker and vals:
                    updates.append({"topic": tw, "value": vals[0], "seq": e["seq"],
                                    "ts": e.get("ts"), "sentence": sent, "role": e.get("role", "")})
                if has_neg:
                    # 否定句（不要求带值）："never integrated X" 是矛盾题的否定侧
                    conflicts.append({"topic": tw, "neg": has_neg, "value": vals,
                                      "seq": e["seq"], "ts": e.get("ts"), "sentence": sent,
                                      "role": e.get("role", "")})
                elif vals and len(tw) >= 3:
                    # 肯定侧池：带值的实义句（矛盾对正侧，marker 放宽）
                    assertives.append({"topic": tw, "value": vals, "seq": e["seq"],
                                       "ts": e.get("ts"), "sentence": sent,
                                       "role": e.get("role", "")})
    return updates, conflicts, assertives


class UpdateConflictIndex:
    """user 级更新链/矛盾对索引（挂在 bucket 上，Add 时增量维护）。"""

    def __init__(self) -> None:
        self.chains: list[dict] = []      # 更新链: {topic, events: [(seq, ts, value, sentence, role)]}
        self.neg_sides: list[dict] = []   # 否定侧: {topic, seq, ts, sentence, role}
        self.assertives: list[dict] = []  # 肯定侧池（矛盾对正侧，marker 放宽）
        self.conflict_pairs: list[dict] = []  # 成型矛盾对
        self.scanned: int = -1            # 已扫描到的 seq 水位（增量防重扫）
        self._pairs_dirty: bool = True    # 矛盾对惰性构建标记（Add 只积累，Search 才构建）

    @staticmethod
    def _overlap(a: list[str], b: list[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / min(len(sa), len(sb))

    def add_updates(self, events: list[dict]) -> None:
        for ev in events:
            chain = max((c for c in self.chains if self._overlap(c["topic"], ev["topic"]) >= 0.5),
                        key=lambda c: self._overlap(c["topic"], ev["topic"]), default=None)
            if chain is None:
                self.chains.append({"topic": ev["topic"],
                                    "events": [(ev["seq"], ev["ts"], ev["value"], ev["sentence"], ev["role"])]})
            else:
                chain["events"].append((ev["seq"], ev["ts"], ev["value"], ev["sentence"], ev["role"]))

    def add_neg_sides(self, events: list[dict]) -> None:
        self.neg_sides.extend(events)
        self._pairs_dirty = True

    def add_assertives(self, events: list[dict]) -> None:
        self.assertives.extend(events)
        self._pairs_dirty = True

    def ensure_pairs(self) -> None:
        """矛盾对惰性构建（Search 时调用，dirty 才重建）。

        Add 路径零成本（官方 Full 实测教训：O(negs×asserts) 全量配对在每次
        Add 重跑，add 耗时超线性增长 37ms→600ms+，3050 条 11 分钟没灌完，
        官方判 ADD_RUNTIME_ERROR）。配对用词倒排加速：每个 neg 只与共享词
        的 asserts 求交，近似线性。"""
        if not self._pairs_dirty:
            return
        # 肯定侧词倒排
        from collections import defaultdict as _dd
        inv: dict[str, list[int]] = _dd(list)
        for idx, a in enumerate(self.assertives):
            for w in set(a["topic"]):
                inv[w].append(idx)
        self.conflict_pairs = []
        for ev in self.neg_sides:
            chain = max((c for c in self.chains if self._overlap(c["topic"], ev["topic"]) >= 0.5),
                        key=lambda c: self._overlap(c["topic"], ev["topic"]), default=None)
            if chain is not None:
                latest = chain["events"][-1]
                self.conflict_pairs.append({
                    "topic": ev["topic"],
                    "side_neg": ev["sentence"], "side_neg_ts": ev["ts"],
                    "side_pos": latest[3], "side_pos_ts": latest[1],
                })
                continue
            cand: dict[int, int] = _dd(int)
            ev_words = set(ev["topic"])
            for w in ev_words:
                for idx in inv.get(w, ()):
                    cand[idx] += 1
            best, best_ov, best_shared = None, 0.0, 0
            for idx, shared in cand.items():
                a = self.assertives[idx]
                if a["seq"] == ev["seq"]:
                    continue
                ov = shared / min(len(a["topic"]), len(ev["topic"]))
                if ov > best_ov:
                    best_ov, best, best_shared = ov, a, shared
            # 短主题句复合标准：共享 ≥2 词且 overlap ≥0.3
            if best is not None and best_shared >= 2 and best_ov >= 0.3:
                self.conflict_pairs.append({
                    "topic": ev["topic"],
                    "side_neg": ev["sentence"], "side_neg_ts": ev["ts"],
                    "side_pos": best["sentence"], "side_pos_ts": best["ts"],
                })
        self._pairs_dirty = False

    def latest_for(self, query_words: set[str], query: str = "") -> dict | None:
        """注入 v2 三重触发（2026-09-24）：数值期望词 + 重叠≥3 + 值类型匹配。

        v1 教训（BEAM -2.6/LME multi -6.0/pmem health -7.3）：重叠≥2 触发太松，
        误注入率 20-50% 把非思考 answer 带偏；ku 目标题型 +7.6/+10.3 证明方向对。
        """
        if not _has_value_expectation(query):
            return None
        best, best_ov = None, 0
        for c in self.chains:
            ov = len(query_words & set(c["topic"]))
            if ov > best_ov:
                best_ov, best = ov, c
        if best is None or best_ov < 3:
            return None
        seq, ts, value, sentence, role = max(best["events"], key=lambda x: x[0])
        return {"seq": seq, "ts": ts, "value": value, "sentence": sentence, "role": role}

    def conflict_for(self, query_words: set[str]) -> dict | None:
        """渲染门控（9-28 收紧）：重叠 ≥4。矛盾对池放宽（肯定侧无 marker）
        后误触发爆炸（BEAM 56% 题带卡）——池可以宽，渲染必须严：
        假阴性退基线零代价，假阳性注入矛盾卡直接带偏 answer。"""
        self.ensure_pairs()
        best, best_ov = None, 0
        for p in self.conflict_pairs:
            ov = len(query_words & set(p["topic"]))
            if ov > best_ov:
                best_ov, best = ov, p
        return best if best_ov >= 4 else None

    def to_dict(self) -> dict:
        return {"chains": self.chains, "conflict_pairs": self.conflict_pairs}

    @classmethod
    def from_dict(cls, d: dict) -> "UpdateConflictIndex":
        idx = cls()
        idx.chains = d.get("chains", [])
        idx.conflict_pairs = d.get("conflict_pairs", [])
        return idx


def latest_record_content(latest: dict) -> str:
    """[LATEST VALUE RECORD] 条目文本 = 原文句 + 日期标注（非 LLM 改写）。"""
    return (f"[LATEST VALUE RECORD | most recent statement, {_fmt_ts(latest['ts'])}] "
            f"{latest['sentence']} (most up-to-date value: {latest['value']})")


def conflict_record_content(pair: dict) -> str:
    """[CONFLICT RECORD] 矛盾对并排呈现 + 行为引导。

    BEAM gold 格式铁律（9-28 badcase 实证）：矛盾题正确答案=指出矛盾并复述
    两侧（"I notice you've mentioned contradictory information... You said X, "
    "but you also mentioned Y"），直接选边答被判错——卡文本显式教 answer 模型
    指出矛盾而非选边。"""
    return (f"[CONFLICT RECORD | contradictory statements exist in memory] "
            f"(a) \"{pair['side_neg']}\" [{_fmt_ts(pair['side_neg_ts'])}]; "
            f"(b) \"{pair['side_pos']}\" [{_fmt_ts(pair['side_pos_ts'])}]. "
            f"These two statements contradict each other — a correct answer must "
            f"acknowledge the contradiction and cite both sides, not pick one.")
