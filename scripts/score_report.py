# -*- coding: utf-8 -*-
"""基线深度分析报告：分类别/分对话/分答案类型 accuracy + 错题导出。

用法: venv/Scripts/python.exe scripts/score_report.py <eval_input.jsonl> <judged.jsonl>
产出: outputs/report_<stamp>.md + outputs/errors_<stamp>.jsonl
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TIME_PAT = re.compile(
    r"\b(20\d{2}|January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|yesterday|today|tomorrow|ago|last|next)\b", re.I)
REL_PAT = re.compile(r"\b(yesterday|today|tomorrow|last\s+\w+|next\s+\w+|\w+\s+ago)\b", re.I)


def main() -> None:
    input_path, judged_path = sys.argv[1], sys.argv[2]
    stamp = Path(judged_path).stem.replace(".judged", "")
    items = {r["id"]: r for r in map(json.loads, open(input_path, encoding="utf-8"))}
    answers = {}
    ans_path = Path(judged_path).with_name(Path(judged_path).name.replace("judged", "answers"))
    if ans_path.exists():
        answers = {r["id"]: r.get("generated_answer", "") for r in map(json.loads, open(ans_path, encoding="utf-8"))}

    by_cat = defaultdict(lambda: [0, 0])
    by_conv = defaultdict(lambda: [0, 0])
    by_time = defaultdict(lambda: [0, 0])  # 答案含时间 / 不含
    by_rel = defaultdict(lambda: [0, 0])   # 预测用了相对时间(可能失分模式)
    errors = []

    for line in open(judged_path, encoding="utf-8"):
        r = json.loads(line)
        item = items.get(r["id"], {})
        ok = int(bool(r.get("is_correct")))
        cat = str(item.get("category", "?"))
        conv = r["id"].split("#")[0]
        gold = " | ".join(item.get("gold_answer", [])) if isinstance(item.get("gold_answer"), list) else str(item.get("gold_answer", ""))
        pred = answers.get(r["id"], "")

        by_cat[cat][0] += ok; by_cat[cat][1] += 1
        by_conv[conv][0] += ok; by_conv[conv][1] += 1
        t = "答案含时间" if TIME_PAT.search(gold) else "普通答案"
        by_time[t][0] += ok; by_time[t][1] += 1
        rel = "预测含相对时间" if (REL_PAT.search(pred) and not REL_PAT.search(gold)) else "正常"
        by_rel[rel][0] += ok; by_rel[rel][1] += 1

        if not ok:
            errors.append({
                "id": r["id"], "category": cat, "question": item.get("question", ""),
                "gold": gold, "pred": pred[:300], "judge_reason": r.get("judge_response", "")[:200],
            })

    total_ok = sum(v[0] for v in by_cat.values())
    total_n = sum(v[1] for v in by_cat.values())
    lines = [f"# 基线报告 {stamp}", "", f"**TOTAL: {total_ok}/{total_n} = {total_ok/total_n*100:.1f}%**", "",
             "## 分类别", "| cat | acc | n |", "|---|---|---|"]
    for cat in sorted(by_cat):
        ok, n = by_cat[cat]
        lines.append(f"| {cat} | {ok/n*100:.1f}% | {n} |")
    lines += ["", "## 分对话", "| conv | acc | n |", "|---|---|---|"]
    for conv in sorted(by_conv, key=lambda c: by_conv[c][0] / by_conv[c][1]):
        ok, n = by_conv[conv]
        lines.append(f"| {conv} | {ok/n*100:.1f}% | {n} |")
    lines += ["", "## 分答案类型", "| 类型 | acc | n |", "|---|---|---|"]
    for t in sorted(by_time):
        ok, n = by_time[t]
        lines.append(f"| {t} | {ok/n*100:.1f}% | {n} |")
    lines += ["", "## 失分模式：预测用了相对时间而gold是绝对时间", "| 类型 | acc | n |", "|---|---|---|"]
    for t in sorted(by_rel):
        ok, n = by_rel[t]
        lines.append(f"| {t} | {ok/n*100:.1f}% | {n} |")
    lines += ["", f"错题共 {len(errors)} 道，明细见 errors_{stamp}.jsonl"]

    out_md = Path(f"outputs/report_{stamp}.md")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    Path(f"outputs/errors_{stamp}.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in errors), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[written] {out_md}")


if __name__ == "__main__":
    main()
