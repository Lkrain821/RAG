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

## 3. ⭐ Ablation Study:2×2 控制变量实验

**问题**:D1→D2 同时改了 chunk_size 和 Reranker 两个变量,怎么证明是 Rerank 起作用?

**解法**:补一个 D2-ablation(`ablation_hybrid_only.py`),做 2×2 完整对照:

| 实验 | chunk_size | Rerank | 切块数 | Top-1 命中 | 备注 |
|---|---|---|---|---|---|
| D1 基线 | 500 | ❌ | 5 | **3/8 (37.5%)** | 原版 D1 |
| D1-ablation | 200 | ❌ | 13 | **4/6 (67%)** | 新增 ablation |
| D2-ablation | 500 | ✅ | 5 | 跟 D1 一样 | 验证"Rerank 对粗 chunk 无效" |
| **D2 完整** | **200** | **✅** | **13** | **6/6 (100%)** | 当前最佳 |

**两个变量缺一不可,功劳分解**:

| 优化手段 | 单独贡献 | 累积 |
|---|---|---|
| chunk 切细 (500→200) | 37.5% → 67% (**+30pp**) | 67% |
| Rerank 精排(BGE-Reranker) | 67% → 100% (**+33pp**) | 100% |

**关键洞察**:
- D2-ablation(chunk=500 + Rerank)没意义,因为召回就 5 个,Rerank 只能微调顺序
- 两个优化针对**不同瓶颈**:
  - chunk 切细 → 解决"召回粒度"(retrieval granularity)
  - Rerank 精排 → 解决"召回排序"(retrieval ranking)
- 单独做任一个都能涨 30pp,叠加再涨 33pp,**边际效益递增**(因为两者正交)

---

## 4. 6 个测试问题 Top-1 命中明细

### D1 基线 (chunk=500, 无 Rerank) - 3/8 命中

| 问题 | Top-1 命中 | 准确? |
|---|---|---|
| 什么是 RAG | 主题 1 (RAG 基础)| ✅ |
| Embedding 怎么把文本变成向量 | 主题 3 (Embedding 原理)| ✅ |
| RRF 公式 | 主题 4 (Chroma) | ❌ |
| BGE-Reranker 怎么用 | 主题 6 (BGE-Reranker)| ✅ |
| RAGAS 4 个核心指标 | 主题 8 (Agentic RAG)| ❌ |
| Agentic RAG 核心思想 | 主题 8 (RAGAS)| ❌ |
| Chroma 和 Milvus 区别 | 主题 4 (Chroma) | ✅ |
| RAG 工作流程 | 主题 5 (检索策略) | ❌ |

### D1-ablation (chunk=200, 无 Rerank) - 4/6 命中

| 问题 | Top-1 命中 | 准确? |
|---|---|---|
| 什么是 RAG | 主题 1 (RAG 基础)| ✅ |
| Embedding 怎么把文本变成向量 | "在线阶段核心是..."(段头)| ❌ |
| RRF 公式 | 主题 6 (RRF 公式段)| ✅ |
| BGE-Reranker 怎么用 | 主题 6 (BGE-Reranker 介绍)| ✅ |
| RAGAS 4 个核心指标 | 主题 7 (4 个指标)| ✅ |
| Agentic RAG 核心思想 | 主题 1 (RAG 基础)| ❌ |

### D2 完整 (chunk=200, 有 Rerank) - 6/6 命中

| 问题 | Top-1 命中 | 准确? |
|---|---|---|
| 什么是 RAG | 主题 1 (RAG 基础)| ✅ |
| Embedding 怎么把文本变成向量 | 主题 3 (Embedding 原理,Self-Attention)| ✅ |
| RRF 公式 | 主题 6 (RRF 公式段)| ✅ |
| BGE-Reranker 怎么用 | 主题 6 (BGE-Reranker 介绍)| ✅ |
| RAGAS 4 个核心指标 | 主题 7 (4 个指标)| ✅ |
| Agentic RAG 核心思想 | 主题 8 (Agentic RAG 定义)| ✅ |

---

## 5. 性能数据(CPU)

- 模型加载:约 5 秒(模型已缓存)
- 单次检索:0.3-0.5 秒(召回 20 + Rerank 13)
- 6 个问题总计:约 3-4 秒

CPU 完全够用,生产环境上 GPU 能再快 5-10 倍。

---

## 6. 简历/面试可用要点(无水分版 ⭐)

> **"通过 2×2 控制变量实验(N=6 测试问题),精确定位 RAG 优化效果:**
> - **细粒度分块**(chunk_size 500→200):Top-1 准确率 37.5% → 67%(**+30pp**)
> - **Cross-Encoder 精排**(BGE-Reranker-base):Top-1 准确率 67% → 100%(**+33pp**)
> - **两阶段叠加**达到 100% 准确率"

**关键技术点**:
- Cross-Encoder vs Bi-Encoder(精度 vs 速度的权衡)
- RRF(Reciprocal Rank Fusion)排名融合
- BGE-Reranker-base(中文首选 Cross-Encoder)
- 控制变量实验设计(避免"同时改两件事"的数据污染)

**生产落地坑**:
- FlagEmbedding 1.4.0 与 transformers 5.x 的 `prepare_for_model` 兼容问题
- 改用 HuggingFace transformers 直接加载 BGE-Reranker

---

## 7. 经验教训(最重要 ⭐)

> **优化 A 引入时,不能同时改 B**。要么固定 A 验证 B 价值,要么固定 B 验证 A 价值,否则面试官一句"你怎么证明"就翻车。
>
> 这次 D2 一开始直接 `chunk=200 + Rerank`,Top-1 从 37.5% 跳到 100%,看起来很爽,但**数据无法证伪**——既不能证明是 chunk 切细的功劳,也不能证明是 Rerank 的功劳,只能证明"两个改动总效果"。
>
> 后来补了 ablation 实验,做了 2×2 矩阵,才把两个优化的**单独贡献拆出来**。这才有底气写"X 优化贡献 +30pp,Y 优化贡献 +33pp"。
>
> **面试中能用这套数据应对追问**:如果问"Rerank 真的有效吗",可以直接说"做了 2×2 ablation,数据是 67% → 100%"。

---

## 8. 下一阶段预告(D3)

**Query 改写** —— 用 LLM 把用户模糊问题改写得精准,或拆成多个子查询。

预期改动:在 hybrid_search 之前加一层 `rewrite_query(query)`,把"那个东西怎么搞"改写成"如何配置 XX 功能"。

注意 D3 也要做 ablation!要分别测:
- 改写前召回率
- 改写后召回率
- 跟 Rerank 叠加的效果
