# -*- coding: utf-8 -*-
"""LoCoMo-Refined 数据集分析：规模、类别、时序占比、AML Add 分块模拟。

用法: venv/Scripts/python.exe scripts/analyze_locomo.py
数据: E:/memoryboard/LoCoMo_refined/data/public/{conversations,questions}.jsonl
"""
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("E:/memoryboard/LoCoMo_refined/data/public")

# ---------- 加载 ----------
convs = [json.loads(l) for l in (ROOT / "conversations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
qs = [json.loads(l) for l in (ROOT / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

print(f"会话数: {len(convs)}   问题数: {len(qs)}")

# ---------- 问题字段探查 ----------
print("\n== 问题字段 ==", sorted(qs[0].keys()))

# ---------- 类别分布 ----------
cat_counts = Counter(q["category"] for q in qs)
print("\n== 类别分布 ==")
for c in sorted(cat_counts):
    print(f"  category {c}: {cat_counts[c]:5d}  ({cat_counts[c]/len(qs)*100:.1f}%)")

# 每类别样例（判断类别语义）
print("\n== 各类别样例（前2条）==")
for c in sorted(cat_counts):
    samples = [q for q in qs if q["category"] == c][:2]
    for s in samples:
        print(f"  [cat {c}] Q: {s['question'][:80]}")
        print(f"          A: {s['answer']}")

# ---------- 答案类型：时间/数字/列表（时序卡点量化） ----------
TIME_PAT = re.compile(r"\b(20\d{2}|January|February|March|April|May|June|July|August|September|October|November|December|"
                      r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|yesterday|today|tomorrow|ago|last|next)\b", re.I)
NUM_PAT = re.compile(r"\d")

n_time, n_num, n_multi_fact = 0, 0, 0
for q in qs:
    ans = " ".join(q["answer"]) if isinstance(q["answer"], list) else str(q["answer"])
    if TIME_PAT.search(ans):
        n_time += 1
    if NUM_PAT.search(ans):
        n_num += 1
    if isinstance(q["answer"], list) and len(q["answer"]) > 1:
        n_multi_fact += 1
print(f"\n== 答案类型占比 ==")
print(f"  答案含时间表达: {n_time} ({n_time/len(qs)*100:.1f}%)  <- 时序推理卡点规模")
print(f"  答案含数字:     {n_num} ({n_num/len(qs)*100:.1f}%)")
print(f"  多候选答案:     {n_multi_fact} ({n_multi_fact/len(qs)*100:.1f}%)")

# ---------- 证据跨度：多跳需要跨 session 的比例 ----------
cross_session = 0
evid_counts = []
for q in qs:
    sess = {m.get("session_index") for m in q.get("evidence_messages", [])}
    evid_counts.append(len(q.get("evidence_messages", [])))
    if len(sess) > 1:
        cross_session += 1
print(f"\n== 证据跨度 ==")
print(f"  证据跨多个 session 的问题: {cross_session} ({cross_session/len(qs)*100:.1f}%)  <- 多跳卡点规模")
print(f"  平均证据条数: {statistics.mean(evid_counts):.2f}  最大: {max(evid_counts)}")

# ---------- 对话规模 ----------
sess_counts, msg_words, conv_words = [], [], []
for conv in convs:
    sessions = conv.get("sessions") or conv.get("conversation") or []
    n_sess = len(sessions) if isinstance(sessions, list) else len(sessions.keys()) if isinstance(sessions, dict) else 0
    sess_counts.append(n_sess)
    total = 0
    msgs = conv.get("messages") or []
    if not msgs and isinstance(sessions, list):
        msgs = [m for s in sessions for m in (s.get("messages", []) if isinstance(s, dict) else [])]
    for m in msgs:
        w = len(str(m.get("text", "")).split())
        msg_words.append(w)
        total += w
    conv_words.append(total)

print(f"\n== 对话规模 ==")
print(f"  每会话 session 数: 均值 {statistics.mean(sess_counts):.1f}  范围 {min(sess_counts)}~{max(sess_counts)}")
print(f"  每对话总词数:     均值 {statistics.mean(conv_words):.0f}  范围 {min(conv_words)}~{max(conv_words)}")
print(f"  总词量(全部对话): {sum(conv_words):,} 词  ≈{int(sum(conv_words)*1.4):,} tokens")

# ---------- AML Add 分块规则模拟 ----------
# 平台规则: 每来源会话一次 Add; >20 条消息或 >2000 词时在最近完整消息/句子边界分段
# 平台已经切好 chunk 发给我们 -> 我们内部可再决定存储粒度。这里模拟平台侧切法。
print(f"\n== AML Add 分块模拟（20条消息/2000词阈值，平台侧切法）==")
total_chunks = []
for conv in convs:
    sessions = conv.get("sessions") or []
    msgs = conv.get("messages") or []
    if not msgs and isinstance(sessions, list):
        msgs = [m for s in sessions for m in (s.get("messages", []) if isinstance(s, dict) else [])]
    # 平台按"来源会话"调用 Add: 先按 session 分组，组内再按 20msg/2000word 切
    by_sess = {}
    for m in msgs:
        by_sess.setdefault(m.get("session_index", 0), []).append(m)
    n_chunks = 0
    for sess_msgs in by_sess.values():
        cur, cur_words = [], 0
        for m in sess_msgs:
            w = len(str(m.get("text", "")).split())
            if (len(cur) >= 20 and cur) or (cur_words + w > 2000 and cur):
                n_chunks += 1
                cur, cur_words = [], 0
            cur.append(m)
            cur_words += w
        if cur:
            n_chunks += 1
    total_chunks.append(n_chunks)
print(f"  每对话 Add chunk 数: 均值 {statistics.mean(total_chunks):.1f}  范围 {min(total_chunks)}~{max(total_chunks)}")
print(f"  全数据集 chunk 总数: {sum(total_chunks)}")
print(f"  → 平均每 chunk ≈ {int(sum(conv_words)/max(1,sum(total_chunks)))} 词")

# ---------- 消息长度分布 ----------
print(f"\n== 消息长度 ==")
print(f"  均值 {statistics.mean(msg_words):.1f} 词, 中位数 {statistics.median(msg_words)}, p95 {sorted(msg_words)[int(len(msg_words)*0.95)]}")

# ---------- 多模态 ----------
mm = sum(1 for q in qs if q.get("is_multi_modality"))
print(f"\n== 多模态问题: {mm} ({mm/len(qs)*100:.1f}%) ==")
