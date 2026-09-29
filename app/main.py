# -*- coding: utf-8 -*-
"""AML 参赛系统入口：严格实现官方 Add / Search / Health 契约。

启动:
  uvicorn app.main:app --host 0.0.0.0 --port 8000

环境变量:
  GRANULARITY     chunk|message            存储粒度（默认 chunk，本地最优 v2chunk）
  SEED_TOP_N      融合排名种子数（默认 20，InvMem 同款）
  NEIGHBOR_SPAN   邻居窗口半径（默认 1，InvMem 同款）

契约要点（与官方文档逐条对应）:
- Add   同步: 写入完成且立即可检索后才返回 200; success=true; 三个 ID 原样返回
- Search data 数组按相关性降序; 不超 top_k; content 直接喂给平台 Answer 模型
- user_id 是唯一检索隔离边界
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field

from .store_v2 import MessageMemoryStore

GRANULARITY = os.environ.get("GRANULARITY", "chunk")
SEED_TOP_N = int(os.environ.get("SEED_TOP_N", "20"))
NEIGHBOR_SPAN = int(os.environ.get("NEIGHBOR_SPAN", "1"))
API_TOKEN = os.environ.get("AML_API_TOKEN", "")  # 空=无认证（本地调试）；线上必设

app = FastAPI(title="AML minimal memory system", version="0.2.0")
store = MessageMemoryStore(seed_top_n=SEED_TOP_N, neighbor_span=NEIGHBOR_SPAN, granularity=GRANULARITY)


async def verify_token(authorization: str = Header(default="")) -> None:
    """官网表单认证方式=Authorization: Token <key>；/health 免认证供探活。"""
    if not API_TOKEN:
        return
    if authorization != f"Token {API_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


# ---------- 请求/响应模型（对齐官方 schema） ----------
class Message(BaseModel):
    role: str
    content: str
    timestamp: int | None = None


class AddRequest(BaseModel):
    request_id: str
    messages: list[Message]
    user_id: str
    session_id: str


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str
    options: list[str] | None = None
    user_id: str
    top_k: int = 100


class MemoryItem(BaseModel):
    id: str
    content: str
    score: float | None = None
    created_at: str | None = None


class SearchResponse(BaseModel):
    data: list[MemoryItem] = Field(default_factory=list)


# ---------- 契约端点 ----------
@app.post("/add", response_model=AddResponse, dependencies=[Depends(verify_token)])
def add(req: AddRequest) -> AddResponse:
    import time as _t
    _t0 = _t.perf_counter()
    messages = [m.model_dump() for m in req.messages]
    store.add(req.user_id, req.session_id, messages, request_id=req.request_id)
    print(f"[req] ADD user={req.user_id[:24]} n={len(messages)} "
          f"{(_t.perf_counter()-_t0)*1000:.0f}ms", flush=True)
    return AddResponse(request_id=req.request_id, user_id=req.user_id, session_id=req.session_id)


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_token)])
def search(req: SearchRequest) -> SearchResponse:
    import time as _t
    _t0 = _t.perf_counter()
    items = store.search(req.user_id, req.query, top_k=req.top_k, options=req.options)
    print(f"[req] SEARCH user={req.user_id[:24]} k={req.top_k} "
          f"ret={len(items)} {(_t.perf_counter()-_t0)*1000:.0f}ms", flush=True)
    return SearchResponse(data=[MemoryItem(**i) for i in items])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---------- 调试端点（非契约，评测不调用） ----------
@app.get("/debug/stats")
def stats() -> dict:
    return store.stats()
