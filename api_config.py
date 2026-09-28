"""Credential-free configuration adapter for the public evaluation pipelines.

与官方 agent-memory-leaderboard/api_config.py 保持一致：
answer/judge 的端点与模型全部来自环境变量（run_pipeline.sh 会先 source .env）。
"""
from __future__ import annotations

import os


ANSWER_API_BASE = os.environ.get("ANSWER_API_BASE", "").rstrip("/")
ANSWER_API_KEY = os.environ.get("ANSWER_API_KEY", "")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "")

JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", "").rstrip("/")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")
JUDGE_VERSION = os.environ.get("JUDGE_VERSION", "")
