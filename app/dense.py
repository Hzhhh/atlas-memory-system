# -*- coding: utf-8 -*-
"""dense 通道：bge-small-en-v1.5 本地向量（InvMem 第 1 名同款）。

设计要点（对齐 InvMem 冻结配置）:
- 文档侧编码裸 content（不加 speaker/时间戳前缀），查询侧加 BGE 英文指令前缀
- normalize_embeddings=True → 余弦 = 点积
- 惰性加载 + 惰性编码（与 store_v2 的 BM25 同生命周期：Add 置脏，Search 重建）
- 任何失败（模型缺失/加载异常）返回 None → 调用方降级 BM25 单路，查询永不失败
"""
from __future__ import annotations

import os
import threading

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_lock = threading.Lock()
_model = None          # SentenceTransformer 单例
_failed = False        # 加载失败标记（只试一次，避免每题重复报错）


def _model_path() -> str:
    # 默认 HF id（任何机器可解析：本地缓存或下载）；本地/服务器路径一律走 AML_DENSE_MODEL 显式注入
    # 教训：曾默认 Windows 硬路径，服务器上被当 repo id 静默禁用 dense（v2_run/self_run 口径事故）
    return os.environ.get("AML_DENSE_MODEL", "BAAI/bge-small-en-v1.5")


def _get_model():
    global _model, _failed
    if _model is not None or _failed:
        return _model
    with _lock:
        if _model is not None or _failed:
            return _model
        try:
            from sentence_transformers import SentenceTransformer
            device = os.environ.get("AML_DENSE_DEVICE", "cpu")
            _model = SentenceTransformer(_model_path(), device=device)
        except Exception as exc:  # 模型缺失/依赖未装
            print(f"[dense] disabled: {exc}")
            _failed = True
    return _model


def encode_docs(texts: list[str], batch: int = 64) -> np.ndarray | None:
    """文档侧批量编码（裸文本，无前缀），返回 (n, dim) L2 归一化矩阵；失败返回 None。"""
    model = _get_model()
    if model is None or not texts:
        return None
    try:
        vecs = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)
    except Exception as exc:
        print(f"[dense] encode_docs failed: {exc}")
        return None


def encode_query(query: str) -> np.ndarray | None:
    """查询侧编码（BGE 英文指令前缀），返回 (dim,) 归一化向量；失败返回 None。"""
    model = _get_model()
    if model is None or not query.strip():
        return None
    try:
        vec = model.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vec[0], dtype=np.float32)
    except Exception as exc:
        print(f"[dense] encode_query failed: {exc}")
        return None


def available() -> bool:
    return _get_model() is not None
