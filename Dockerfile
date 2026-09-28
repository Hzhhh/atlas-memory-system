FROM python:3.11-slim

WORKDIR /srv/aml

# CPU 版 torch（默认 CUDA 版 2GB+，CPU 版 ~200MB，ECS 无 GPU）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预烤 bge-small-en-v1.5 到固定目录（运行时离线加载，评测不依赖 HF 网络）
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-small-en-v1.5').save('/srv/aml/models/bge-small')
EOF

COPY app ./app

# 统一配置（9.30 提交版）：单一系统，不感知数据集
# - GRANULARITY=message：官方 Add 分段天然粒度，防 117K 窗口保序截断丢长尾
# - dense+TIME+ENTITY+注入v2 全开（= store_v2 默认值，显式声明防意外）
# - PERSONA=0：Add 端模型合规（开源榜 Add 必须 gpt-4o-mini；persona 由 qwen 生成不合规）
# - ABSTAIN=0：拒答截断未验证，首提不开
ENV GRANULARITY=message \
    SEED_TOP_N=20 \
    NEIGHBOR_SPAN=1 \
    AML_DENSE=1 \
    AML_INJECT=1 \
    AML_TIME=1 \
    AML_ENTITY=1 \
    AML_PERSONA=0 \
    AML_ABSTAIN=0 \
    AML_TIMELINE=0 \
    AML_API_TOKEN=CHANGE_ME_AT_DEPLOY \
    AML_DENSE_MODEL=/srv/aml/models/bge-small \
    AML_DENSE_DEVICE=cpu \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
