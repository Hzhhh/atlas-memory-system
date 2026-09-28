# AML Memory System (Rule-based Retrieval Engineering)

Agent Memory Leaderboard (AML) 第二期文本赛道参赛系统 —— 学术开源组。

纯规则检索工程：**Add/Search 端零 LLM**，所有增强组件（更新链检测、矛盾对索引、实体倒排、时间戳前缀、相对时间解析）均为规则实现。

## 方法

### 检索底座
- **混合检索**：BM25 + dense（bge-small-en-v1.5）加权 RRF 融合（dense 1.0 / BM25 0.5 / k=60）
- **种子 + 邻居扩展**：融合排名 top-20 种子，每个种子拉取同 session 相邻 ±1 条消息
- **实体两阶段扩展**：种子命中的实体经倒排索引自查补全（列表完整性）
- **索引键增强**：说话人 + 事件日期前缀进 BM25 语料；返回条目带 `[mem-{id} | {role} | {date}]` 前缀与相对时间解析注解

### 记忆增强（注入组件）
- **[LATEST VALUE RECORD]**：Add 端更新链检测（marker 词 + 值抽取），值期望类查询触发注入最新值原文句（三重门控：期望词元 + 主题重叠 ≥3 + 值类型匹配）
- **[CONFLICT RECORD]**：否定侧 × 肯定侧池主题配对的矛盾对索引，渲染门控 ≥4 词重叠

### 工程要点
- dense 惰性增量编码（缓存 + vstack，与全量编码比特级一致）
- 惰性索引与 Add 置脏机制（BM25 全量重建 / dense 增量）
- 单一系统单一配置，不感知数据集身份

## API

```
POST /add      {request_id, user_id, session_id, messages: [{role, content, timestamp}]}
POST /search   {query, user_id, top_k, options?}
GET  /health
```

认证：`Authorization: Token <key>`（/health 免认证）。

## 部署

```bash
docker build -t aml-system .
docker run -d -p 8000:8000 -e AML_API_TOKEN=<your-token> aml-system
```

依赖：bge-small-en-v1.5 模型在镜像构建时预下载（离线加载，评测不依赖外网）。

## 复现

统一配置（即提交配置）由 Dockerfile 环境变量固化。本地调试：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

环境变量开关：`AML_DENSE / AML_INJECT / AML_TIME / AML_ENTITY / AML_ABSTAIN`（提交配置全 1/1/1/1/0）。

## License

MIT
