# D2 - BGE-Reranker 精排 学习笔记

**日期**:2026-06-08
**目标**:在 D1 混合检索基础上加 BGE-Reranker Cross-Encoder 精排

---

## 1. 核心改动

| 维度 | D1 | D2 |
|---|---|---|
| 召回阶段 | BM25 + 向量(EnsembleRetriever) | 同 D1,但 k=5 → **k=20** |
| 精排阶段 | ❌ 无 | ✅ BGE-Reranker-base(Cross-Encoder)|
| chunk_size | 500 | **200** |
| chunk_overlap | 50 | 30 |
| 切块数 | 5 | **13** |
| Top-1 准确率 | 3/8 (37.5%) | **6/6 (100%)** |

---

## 2. 踩过的两个真实坑

### 坑 1:FlagEmbedding 1.4.0 与 transformers 5.x 不兼容

```
AttributeError: XLMRobertaTokenizer has no attribute 'prepare_for_model'
```

`pip install -U FlagEmbedding` 显示 Requirement already satisfied(1.4.0 是最新),但**这个 bug 没修**。

**修法**:绕开 FlagEmbedding,直接用 HuggingFace transformers 加载 BGE-Reranker:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
model.eval()

# 推理:输入 [query, doc] 对,输出相关性分数
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
scores = model(**inputs).logits.view(-1).float()
```

**数学上完全等价** —— BGE-Reranker 本质就是 SequenceClassification,FlagEmbedding 只是封装。

### 坑 2:chunk 切太粗,Rerank 没空间发挥

第一版跑下来,6 个问题的 Top-5 **跟 D1 完全一样**,差点以为 Rerank 没生效。

**根因**:`切了 5 个块`,整个文档就 5 个 chunk,BM25/向量最多召回 5 个,Reranker 拿到 5 个候选做精排 = 给定输入排序,结果跟 RRF 排序一样。

**修法**:`chunk_size=500 → 200`,切块数从 5 变 13,Rerank 真正起作用。

---

## 3. 6 个测试问题 Top-1 命中明细

| 问题 | Top-1 命中 | 是否准确 |
|---|---|---|
| 什么是 RAG | 主题 1(RAG 基础) | ✅ |
| Embedding 怎么把文本变成向量 | 主题 3(Embedding 原理,Self-Attention) | ✅ |
| RRF 公式 | 主题 6(RRF 公式段) | ✅ |
| BGE-Reranker 怎么用 | 主题 6(BGE-Reranker 介绍) | ✅ |
| RAGAS 4 个核心指标 | 主题 7(4 个指标列表) | ✅ |
| Agentic RAG 核心思想 | 主题 8(Agentic RAG 定义) | ✅ |

**6/6 = 100%**(D1 是 3/8 = 37.5%)

---

## 4. 性能数据(CPU)

- 模型加载:约 5 秒(模型已缓存)
- 单次检索:0.3-0.5 秒(召回 20 + Rerank 13)
- 6 个问题总计:约 3-4 秒

CPU 完全够用,生产环境上 GPU 能再快 5-10 倍。

---

## 5. 简历/面试可用要点

> "实现 Advanced RAG 三大件:BM25 + Dense 混合检索、Cross-Encoder 精排、Hybrid Search。
> 在 13 个分块的测试集上,Top-1 准确率从 37.5% 提升到 100%(N=6 个测试问题)。
> 落地中处理了 FlagEmbedding 与 transformers 5.x 的版本兼容问题。"

**量化数据**:
- Top-1 准确率:37.5% → 100%
- 召回:20 个候选 → 精排:13 个有效 chunk

**关键技术点**:
- Cross-Encoder vs Bi-Encoder(精度 vs 速度的权衡)
- RRF(Reciprocal Rank Fusion)排名融合
- BGE-Reranker-base(中文首选 Cross-Encoder)

---

## 6. 下一阶段预告(D3)

**Query 改写** —— 用 LLM 把用户模糊问题改写得精准,或拆成多个子查询。

预期改动:在 hybrid_search 之前加一层 `rewrite_query(query)`,把"那个东西怎么搞"改写成"如何配置 XX 功能"。
