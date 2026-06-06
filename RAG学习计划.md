# RAG 系统学习 30 天计划

> **作者**: Mavis
> **创建日期**: 2026-06-06
> **适用对象**: 已跑通 Naive RAG,具备 Python + LangChain 基础
> **目标产出**: 简历可写 5 个真实项目 + 秋招(7-8 月)技术储备
> **总工时预估**: 约 150 小时(30 天 × 平均 5h/天,周末 8h/天)

---

## 📋 整体路线

```
总周期 30 天 / 5 个阶段 / 每周 5 个工作日 + 周末加量
═══════════════════════════════════════════════════════════

Stage 1 [D1-D3]    Advanced RAG 三大件         ← 决定 RAG 效果上限
   ↓
Stage 2 [D4-D6]    RAG 评估体系                ← 简历杀手锏,大部分人跳过
   ↓
Stage 3 [D7-D9]    语义切块 + Parent-Doc       ← 数据层优化
   ↓
Stage 4 [D10-D17]  Agentic RAG(Self/CRAG)     ← 2024-2025 主战场
   ↓
Stage 5 [D18-D22]  工程化(FastAPI + UI)        ← 落地能力
   ↓
Stage 6 [D23-D30]  前沿 + 简历整合             ← 加分项
```

**为什么这个顺序?**
- **评估优先于优化**:没评估你不知道改动有没有用
- **数据 > 模型**:切块/检索优化比换 LLM 收益大 10 倍
- **Agentic 放后**:你已经有 LangGraph 基础,这是顺势而为
- **工程化最后**:先把模型层做到位,UI 是表面功夫

---

## 🎯 阶段产出与简历映射

每个阶段结束后,简历上能多一个可写的项目 / 技能点:

| Stage | 产出项目 | 简历可写亮点 |
|---|---|---|
| Stage 1 | 升级版 RAG(混合检索 + Rerank + Query 改写) | "Advanced RAG 三件套" |
| Stage 2 | RAGAS 评估报告 | "RAGAS 4 指标,baseline vs 改造后 89% 提升" |
| Stage 3 | 语义切块 + 父块子块检索 | "复杂文档(法律/医疗)处理" |
| Stage 4 | Agentic RAG 知识库 | "Self-Reflection + RAGAS 评估闭环" |
| Stage 5 | FastAPI + Streamlit Web 版 | "端到端 Web 部署" |
| Stage 6 | 简历多版本 + GitHub 整理 | 投递准备 |

---

# Stage 1: Advanced RAG 三大件(D1-D3)

> **本阶段目标**:把现有 Naive RAG 升级到 Advanced RAG,跑通"混合检索 + Rerank + Query 改写"三件套。

## 📚 D1(第 1 天)混合检索:BM25 + 向量,RRF 融合

### 学习目标
- [ ] 理解为什么纯向量检索会漏召回(精确术语 / 编号)
- [ ] 理解 BM25 原理
- [ ] 理解 Reciprocal Rank Fusion(RRF)融合公式
- [ ] 在现有 RAG 项目上实现 BM25 + 向量混合检索

### ⚠️ 重点消化
- **BM25 不是"老古董"** —— 2024 年混合检索仍是工业界标配,BM25 + Dense 各负责不同召回
- **RRF 是融合首选** —— 比加权融合简单且不需要训练,公式:`score = Σ 1/(k + rank_i)`,k 通常取 60
- **何时混合有效**:精确查询(编号/术语/姓名)+ 模糊查询(语义),混合显著好于单策略

### 资料清单
- 📄 [Advanced RAG Techniques: an Illustrated Overview](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8f90d) — 一篇顶十篇
- 📄 [Reciprocal Rank Fusion 论文(2021)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — Cormack 经典
- 🎥 [LangChain 官方 RAG 教程](https://python.langchain.com/docs/tutorials/rag/) — 必看
- 🎥 [BM25 Explained(YouTube)](https://www.youtube.com/results?search_query=bm25+explained) — 5 分钟讲清
- 🔧 [rank_bm25 GitHub](https://github.com/dorianbrown/rank_bm25) — Python 实现

### 任务清单
- [ ] 读 Advanced RAG 综述 **前 1/3**(1.5h)
- [ ] 读 RRF 论文,**只看公式部分**(20min)
- [ ] 在 `D:\Pythoncode\RAG` 新建 `hybrid_retriever.py`(2.5h)
  ```python
  # 伪代码骨架
  from langchain_chroma import Chroma
  from langchain.retrievers import BM25Retriever, EnsembleRetriever

  # 向量检索
  vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
  # BM25
  bm25_retriever = BM25Retriever.from_documents(chunks)
  bm25_retriever.k = 5

  # RRF 融合
  ensemble_retriever = EnsembleRetriever(
      retrievers=[bm25_retriever, vector_retriever],
      weights=[0.5, 0.5]
  )
  ```
- [ ] 用 5 条已知答案的测试问题对比:纯向量 vs 混合,**记录准确率差异**(30min)
- [ ] 写 `D:\Pythoncode\RAG\notes\D1_hybrid_search.md`(30min)

### 产出物
- ✅ `hybrid_retriever.py` 可运行
- ✅ A/B 对比数据(2-3 张表)
- ✅ Git commit:`feat: hybrid retrieval with BM25 + vector + RRF`

### 时间分配
- 阅读 2h + 编码 2.5h + 笔记 0.5h = **5h**

---

## 📚 D2(第 2 天)Rerank:BGE-Reranker 精排

### 学习目标
- [ ] 理解 Bi-Encoder vs Cross-Encoder 区别
- [ ] 理解为什么 Rerank 能显著提升 Top-K 质量
- [ ] 在 RAG 中插入 Rerank 阶段
- [ ] 对比 Rerank 前后 Top-5 准确率

### ⚠️ 重点消化
- **Rerank 必加**:工程经验 —— 任何严肃 RAG 系统都有 Rerank,不加等于浪费 30% 准确率
- **Bi-Encoder 编码,Cross-Encoder 精排**:Bi 速度快但精度低,Cross 速度慢但精度高,Rerank 用 Cross 是标准做法
- **BGE-Reranker 是中文首选** —— [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base),CPU 也能跑

### 资料清单
- 📄 [BGE Reranker 论文(2023)](https://arxiv.org/abs/2309.07597) — 看 Abstract + 1 张架构图
- 📄 [Cross-Encoder 原理解释](https://www.sbert.io/examples/applications/cross-encoder/README.html) — Sentence-Transformers 官方
- 🔧 [FlagEmbedding GitHub](https://github.com/FlagOpen/FlagEmbedding) — 包含 BGE Reranker 实现
- 🎥 [BGE Reranker 教程](https://github.com/FlagOpen/FlagEmbedding/tree/master/tutorials) — 5 分钟跑通

### 任务清单
- [ ] 读 BGE Reranker 论文 **Abstract + 架构图**(20min)
- [ ] 读 SBERT Cross-Encoder 文档(20min)
- [ ] 安装 `FlagEmbedding`:
  ```bash
  pip install FlagEmbedding
  ```
- [ ] 新建 `D:\Pythoncode\RAG\reranker.py`(2h)
  ```python
  from FlagEmbedding import FlagReranker
  reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=False)
  # 给定 query 和 documents,返回精排后的 top_k
  ```
- [ ] 改造 `chat.py`,在 retrieve 后插入 rerank 步骤(2h)
- [ ] 用 5-10 条问题做 A/B 测试:**无 rerank vs 有 rerank**(30min)
- [ ] Git commit + 写笔记(30min)

### 产出物
- ✅ `reranker.py` + 改造后的 `chat.py`
- ✅ Rerank 前后对比表(至少 5 行)
- ✅ 笔记 `notes/D2_rerank.md`

### 时间分配
**5h**

---

## 📚 D3(第 3 天)Query 改写:MultiQuery + HyDE

### 学习目标
- [ ] 理解为什么用户问题常常"搜不到"
- [ ] 掌握 2 种主流改写方法:MultiQueryRetriever、HyDE
- [ ] 实现 Query 改写 + 重查循环
- [ ] 完成 Advanced RAG 三大件,产出 1.0 版本

### ⚠️ 重点消化
- **MultiQuery**:用 LLM 把 1 个问题改成 3-5 个不同视角去检索,简单有效
- **HyDE**:让 LLM "先假想一个答案",用这个假答案的 embedding 去检索,适合"答案风格独特"的问题
- **改写的代价**:每次检索多调 1 次 LLM,延迟 + 成本,生产环境要评估 ROI

### 资料清单
- 📄 [MultiQuery Retriever 文档](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever) — 必看
- 📄 [HyDE 论文(2022)](https://arxiv.org/abs/2212.10496) — Hypothetical Document Embeddings
- 🎥 [LangChain Query Rewriting 教程](https://blog.langchain.dev/query-transformations/)
- 📄 [Step-back Prompting 论文(2023)](https://arxiv.org/abs/2310.06117) — 另一种改写思路

### 任务清单
- [ ] 读 HyDE 论文 **Abstract + 1 个示例**(15min)
- [ ] 读 MultiQuery 文档 + Step-back 论文(30min)
- [ ] 新建 `query_rewriter.py`(2h)
  ```python
  from langchain.retrievers.multi_query import MultiQueryRetriever
  from langchain.retrievers import HydeRetriever  # 自实现
  ```
- [ ] 实现"重查循环":检索 → 评分 → 不够就改写 → 重查,**最多 3 次**(2h)
- [ ] 跑 5 条测试问题,**记录 3 种策略的命中差异**(30min)
- [ ] 写 Stage 1 总结笔记 `notes/STAGE1_SUMMARY.md`(30min)
- [ ] Git commit + push

### 产出物
- ✅ `query_rewriter.py` + 改写循环
- ✅ Stage 1 总结(可作为简历"项目亮点"素材)
- ✅ Git tag:`v1.0-advanced-rag`

### 时间分配
**5h**

---

### 🎁 Stage 1 完结验收
- [ ] 跑 20 条测试问题,记录命中率
- [ ] 准备 5-8 张对比图(可选)
- [ ] 在 README 写一段 "Stage 1 升级效果"

### 💡 简历可写(改写前)
> "实现 Advanced RAG 三大件:BM25 + Dense 混合检索(BGE 精排)、BGE-Reranker Cross-Encoder 精排、MultiQuery + HyDE Query 改写重查,准确率从 X% 提升到 Y%(具体跑分后填)"

---

# Stage 2: RAG 评估体系(D4-D6)

> **本阶段目标**:建立 RAGAS 评估 pipeline,**用数据驱动后续所有优化**。
> **为什么这阶段重要**:面试时拿出 4 个评估指标 + 数字,立刻从"做过 RAG"升级到"懂 RAG"。

## 📚 D4(第 4 天)RAG 评估理论基础

### 学习目标
- [ ] 理解为什么 RAG 评估是难题(无标准答案)
- [ ] 掌握 4 个核心指标定义
- [ ] 理解 LLM-as-Judge 范式

### ⚠️ 重点消化
- **4 个核心指标必须背熟**:
  - `context_precision`:检索内容的"准"(信号噪声比)
  - `context_recall`:应该召回的召回了多少(漏检)
  - `faithfulness`:答案是否完全基于检索(防幻觉)
  - `answer_relevancy`:答案与问题的相关度
- **LLM-as-Judge 是主流**:用 GPT-4 / Claude 评判另一个模型的输出,代替人工标注
- **测试集 ≥ 30 条**:低于 30 条指标不稳,30-100 条是工业界标准

### 资料清单
- 📄 [RAGAS 论文(2023)](https://arxiv.org/abs/2309.15217) — 评估框架事实标准
- 🔧 [RAGAS 官方文档](https://docs.ragas.io/en/stable/) — 必看 introduction
- 📄 [LLM-as-Judge 综述](https://arxiv.org/abs/2311.09727) — 评估范式
- 📄 [Context Precision/Recall 解释](https://docs.ragas.io/en/stable/concepts/metrics/index.html) — 指标定义

### 任务清单
- [ ] 读 RAGAS 论文 **Abstract + 4 个指标定义**(1h)
- [ ] 读 RAGAS 官方 introduction 全部(1h)
- [ ] 安装 RAGAS:
  ```bash
  pip install ragas
  ```
- [ ] 写 `notes/D4_evaluation_theory.md`(1h)
- [ ] 准备 30-50 条测试集:`data/eval_qa.jsonl`(2h)
  - 每条格式:`{"question": "...", "ground_truth": "..."}`
  - 从你现有 PDF / 文档里挑真实问题

### 产出物
- ✅ `data/eval_qa.jsonl` 30-50 条
- ✅ 笔记 `notes/D4_evaluation_theory.md`

### 时间分配
**5h**

---

## 📚 D5(第 5 天)RAGAS 实操 + baseline 跑分

### 学习目标
- [ ] 跑通 RAGAS 4 指标 baseline
- [ ] 准备"前 vs 后"对比框架
- [ ] 理解 RAGAS 在中文场景的注意事项

### ⚠️ 重点消化
- **RAGAS 需要"强 LLM"做 judge**:推荐 GPT-4 / Claude / DeepSeek,本地 Ollama(qwen3:4b)做 judge **不靠谱,会偏高**
- **成本估算**:30 条 × 4 指标 × 每次 judge 调 1 次 = ~120 次 LLM 调用,约 5-10 元
- **Baseline 永远先跑**:你 Stage 1 的"未优化版本"必须先有 baseline,否则改造完不知道有没有效

### 资料清单
- 🔧 [RAGAS Quickstart](https://docs.ragas.io/en/stable/getstarted/quickstart/) — 必看
- 🔧 [RAGAS 中文教程](https://docs.ragas.io/en/stable/howtos/integrations/_langchain/) — LangChain 集成
- 📄 [RAGAS 评估中文实战博客](https://zhuanlan.zhihu.com/p/678885538) — 知乎

### 任务清单
- [ ] 读 RAGAS Quickstart(1h)
- [ ] 注册 DeepSeek API key(已经有)或用 OpenAI(2min)
- [ ] 写 `D:\Pythoncode\RAG\eval.py`(3h)
  ```python
  from ragas import evaluate
  from ragas.metrics import (
      context_precision, context_recall,
      faithfulness, answer_relevancy
  )
  from datasets import Dataset

  # 准备 dataset
  eval_dataset = Dataset.from_dict({
      "question": [...],
      "contexts": [...],
      "answer": [...],
      "ground_truth": [...]
  })

  result = evaluate(eval_dataset, metrics=[...])
  print(result)
  ```
- [ ] 跑 baseline(当前 Naive RAG 状态)(1h)
- [ ] 写 `notes/D5_baseline.md` 记录分数

### 产出物
- ✅ `eval.py` 可运行
- ✅ Baseline 报告:`reports/baseline_metrics.json`

### 时间分配
**6h**

---

## 📚 D6(第 6 天)Stage 1 改造后的评估

### 学习目标
- [ ] 跑改造后(混合检索 + Rerank + Query 改写)的 RAGAS
- [ ] 对比 baseline vs 改造后,生成可视化报告
- [ ] 写 Stage 2 总结

### ⚠️ 重点消化
- **4 个指标**全部要跑,不要只看一个 —— 整体提升可能不大但分项差异明显
- **典型优化模式**:
  - 改写后 `context_recall` 涨(查得更全)
  - Rerank 后 `context_precision` 涨(查得更准)
  - Faithfulness 通常不靠这些改,靠 Prompt 工程
- **报告必须画图**:HR / 面试官最喜欢看图

### 任务清单
- [ ] 改造 `eval.py` 接受 `--mode` 参数:`baseline` / `advanced`(2h)
- [ ] 跑改造后,保存到 `reports/advanced_metrics.json`(1h)
- [ ] 写 `compare.py` 生成对比图(matplotlib)(1.5h)
  - 4 指标 × 2 模式 = 8 个柱状图
  - 或者 1 个综合雷达图
- [ ] 写 `notes/STAGE2_SUMMARY.md`(1h)
- [ ] 把报告截图保存到 `assets/screenshots/`

### 产出物
- ✅ `reports/advanced_metrics.json`
- ✅ `assets/screenshots/ragas_comparison.png`
- ✅ 简历可写素材:"RAGAS 评估,4 指标全面提升 X%"

### 时间分配
**5.5h**

---

### 🎁 Stage 2 完结验收
- [ ] 4 指标 baseline vs 改造后对比图
- [ ] 改进点 3 条总结(可放简历)
- [ ] 把 RAGAS 集成进 `eval.py`,后续每次改动都跑一遍

---

# Stage 3: 语义切块 + Parent-Doc 检索(D7-D9)

> **本阶段目标**:把"切块"从硬编码升级到语义切块 + 父块子块检索。
> **为什么重要**:数据层优化,影响所有上层 RAG 效果。

## 📚 D7(第 7 天)SemanticChunker 语义切块

### 学习目标
- [ ] 理解固定长度切块的缺陷
- [ ] 掌握 LangChain SemanticChunker 原理(基于 embedding 相似度)
- [ ] 在现有 RAG 项目替换切块器

### ⚠️ 重点消化
- **SemanticChunker 原理**:用 embedding 计算相邻句子的相似度,相似度低的地方切一刀
- **不要追求完美切块**:切块只是 RAG 链路的一环,过度优化会浪费 3 天收效甚微
- **何时用 SemanticChunker**:长文档(>5 页 PDF)、结构松散(博客/书);规则文档(合同/代码)用 Recursive 即可

### 资料清单
- 🔧 [LangChain SemanticChunker 文档](https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic_chunker) — 必看
- 📄 [Chunking Strategies for RAG 综述](https://www.pinecone.io/learn/series/rag/rerankers/) — Pinecone 经典
- 🔧 [Greg Kamradt 切块教程](https://github.com/FullStackRetrieval-com/RetrievalTutorials) — GitHub

### 任务清单
- [ ] 读 LangChain SemanticChunker 文档(45min)
- [ ] 准备 2 份不同类型文档:1 份长 PDF + 1 份规则文档(30min)
- [ ] 写 `chunking_experiments.py`,对比 4 种切块策略(2.5h):
  - 固定长度
  - RecursiveCharacterTextSplitter
  - SemanticChunker
  - MarkdownHeaderTextSplitter
- [ ] 用 5 条问题跑评估,记录 4 种策略的指标差异(1.5h)
- [ ] 笔记 `notes/D7_semantic_chunking.md`(30min)

### 时间分配
**6h**

---

## 📚 D8(第 8 天)Parent-Document Retriever

### 学习目标
- [ ] 理解"细检索粗返回"的核心思想
- [ ] 掌握 LangChain ParentDocumentRetriever 用法
- [ ] 实现 2 级索引(子块 200 字 + 父块 1500 字)

### ⚠️ 重点消化
- **核心理念**:检索用细粒度子块(查得准),返回用粗粒度父块(上下文够)
- **典型应用**:法律 / 医疗 / 金融文档,需要"看到完整条款 / 完整病历"
- **存储代价**:需要同时存 2 套(子块和父块),用 docstore 关联

### 资料清单
- 🔧 [LangChain ParentDocumentRetriever 文档](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever) — 必看
- 📄 [RAG 高级检索策略综述](https://blog.langchain.dev/parent-document-retriever/)

### 任务清单
- [ ] 读 ParentDocumentRetriever 文档 + 综述(1h)
- [ ] 写 `parent_doc_retriever.py`(3h)
- [ ] 准备一份 10+ 页的 PDF 文档做实验(30min)
- [ ] 对比:普通 retriever vs ParentDoc retriever(2h)
- [ ] 写笔记(30min)

### 时间分配
**7h**(周末加量)

---

## 📚 D9(第 9 天)切块 + 检索优化总结

### 学习目标
- [ ] 整合 Stage 1-3 成果
- [ ] 跑最终 RAGAS 评估
- [ ] 写 Stage 3 总结

### 任务清单
- [ ] 把语义切块 + ParentDoc 接入主链路(2h)
- [ ] 跑 RAGAS 评估,记录 v2.0 分数(1.5h)
- [ ] 写 `notes/STAGE3_SUMMARY.md`(1h)
- [ ] 整理代码,清理无用文件(1h)
- [ ] Git tag:`v2.0-semantic-chunking`

### 时间分配
**5.5h**

---

# Stage 4: Agentic RAG(D10-D17)— 核心

> **本阶段目标**:用 LangGraph 把现有 RAG 升级为 Agentic RAG,实现 Self-RAG / CRAG / Reflector 三大能力。
> **简历杀手锏**:这是 2024-2025 市场最火的方向,8 天投入 = 直接变成简历亮点。

## 📚 D10(第 10 天)Self-RAG 论文 + LangGraph 复习

### 学习目标
- [ ] 通读 Self-RAG 论文,理解 reflection token 机制
- [ ] 复习 LangGraph StateGraph(你已经会)
- [ ] 设计本项目的 LangGraph 状态机

### ⚠️ 重点消化
- **Self-RAG 核心**:3 种 reflection token — `Retrieve`、`IsRel`(检索相关)、`IsSup`(支持答案)、`IsUse`(答案有用)
- **简化版 Self-RAG**:不必完全复现论文,只需做"IsRel 评分 + IsUse 自评"
- **LangGraph 状态机设计原则**:每个节点做一件事、状态用 Pydantic、边用 Conditional Edges

### 资料清单
- 📄 [Self-RAG 论文(2023)](https://arxiv.org/abs/2310.11511) — 必读
- 🔧 [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) — 复习 StateGraph
- 🔧 [LangGraph 官方 Agentic RAG 教程](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/) — **必看,几乎手把手**
- 🎥 [吴恩达 Agentic RAG 短课](https://www.deeplearning.ai/short-courses/agentic-rag/) — 1h,讲得最清楚

### 任务清单
- [ ] 读 Self-RAG 论文(1.5h)
- [ ] 看完 LangGraph 官方 Agentic RAG 教程(2h)
- [ ] 看完吴恩达 Agentic RAG 短课(1h)
- [ ] 设计本项目状态机:`notes/D10_state_graph_design.md`(1.5h)
  - 节点:Router / Retrieve / Grade / Rewrite / Generate / Reflector
  - 边:Conditional based on grade
  - State:TypedDict 或 Pydantic

### 时间分配
**6h**

---

## 📚 D11(第 11 天)LangGraph 化现有 RAG

### 学习目标
- [ ] 把现有 `retrieve` / `grade` 改造成 LangGraph 节点
- [ ] 实现最小可用状态机:Retrieve → Grade → Generate
- [ ] 跑通端到端

### 任务清单
- [ ] 新建 `agentic_rag/graph.py`(3h)
- [ ] 定义 `State` (TypedDict 或 Pydantic)(30min)
- [ ] 实现 3 个节点:retrieve / grade / generate(2h)
- [ ] 跑通,确保能输出答案(1h)
- [ ] Git commit(15min)

### 时间分配
**7h**

---

## 📚 D12(第 12 天)Router 节点(多路径决策)

### 学习目标
- [ ] 实现 LLM Router,让 Agent 自主选择路径
- [ ] 路径:vectorstore / web_search / ask_user
- [ ] 跑通带 Router 的版本

### ⚠️ 重点消化
- **Router Prompt 要严格** —— 必须限定只返回 3 个单词之一,否则会出格式错误
- **加 Pydantic 结构化输出** 比纯 prompt 稳:`Literal["vectorstore", "web_search", "ask_user"]`

### 任务清单
- [ ] 写 `graph.py` 加 Router 节点(2h)
- [ ] Pydantic 结构化输出(1h)
- [ ] Conditional Edge 分发(1.5h)
- [ ] 测试 10 条问题(2h)
- [ ] Git commit(15min)

### 时间分配
**7h**

---

## 📚 D13(第 13 天)Query Rewriter + 改写重查循环

### 学习目标
- [ ] 实现 Rewriter 节点
- [ ] 改写后重查,**最多 3 次重试**
- [ ] 与 Grader 联动形成闭环

### 任务清单
- [ ] 写 Rewriter 节点(2h)
- [ ] 实现 max_retries 逻辑(1.5h)
- [ ] Conditional Edge:grade 不达标就 rewrite(1h)
- [ ] 测试 10 条问题,统计重查率(2h)
- [ ] Git commit(15min)

### 时间分配
**6.5h**

---

## 📚 D14(第 14 天)Generate + Reflector 节点

### 学习目标
- [ ] 优化 Generate 节点(基于检索内容、引用溯源)
- [ ] 实现 Reflector 节点(答案自评幻觉)
- [ ] 不达标回到 Router 换路径

### 任务清单
- [ ] 重写 Generate Prompt(1.5h)
- [ ] 写 Reflector 节点(2.5h)
- [ ] Conditional Edge:不达标 → Router(1.5h)
- [ ] 测试 10 条问题,统计 Reflector 重答率(2h)
- [ ] Git commit(15min)

### 时间分配
**7.5h**

---

## 📚 D15(第 15 天)集成 + RAGAS 评估 v3.0

### 学习目标
- [ ] 集成 Stage 4 全部能力
- [ ] 跑 RAGAS 评估,产出 v3.0 数据
- [ ] 与 v2.0 对比

### 任务清单
- [ ] 集成测试(2h)
- [ ] 跑 RAGAS 评估(2h)
- [ ] 生成对比报告(1.5h)
- [ ] 写 `notes/STAGE4_INTERIM.md`(1h)
- [ ] Git tag:`v3.0-agentic-rag`

### 时间分配
**6.5h**

---

## 📚 D16(第 16 天)优化 Reflector Prompt + 错误案例分析

### 学习目标
- [ ] 分析 5-10 条 Reflector 误判的 case
- [ ] 优化 Prompt 让 Reflector 更准
- [ ] 减少无意义的重答

### 任务清单
- [ ] 收集错误 case(1.5h)
- [ ] 分析根因(1.5h)
- [ ] 优化 Prompt(2h)
- [ ] 重新跑评估(2h)
- [ ] 笔记(1h)

### 时间分配
**8h**(周末)

---

## 📚 D17(第 17 天)Stage 4 完结总结

### 学习目标
- [ ] 写 Stage 4 总结文档
- [ ] 准备简历项目素材
- [ ] 写 README

### 任务清单
- [ ] 写 `notes/STAGE4_SUMMARY.md`(2h)
- [ ] 整理 Agentic RAG 项目结构(1h)
- [ ] 写完整 README(2h)
- [ ] 准备简历项目文案(1h)
- [ ] Git tag:`v3.1-agentic-rag-final`

### 时间分配
**6h**

---

# Stage 5: 工程化(D18-D22)

> **本阶段目标**:把命令行 RAG 升级为可演示的 Web 应用。
> **简历加分**:能 demo 比能跑通重要 10 倍。

## 📚 D18(第 18 天)FastAPI 后端

### 学习目标
- [ ] 用 FastAPI 暴露 RAG 接口
- [ ] 支持流式响应(SSE)
- [ ] 写 OpenAPI 文档

### 资料清单
- 🔧 [FastAPI 官方文档](https://fastapi.tiangolo.com/zh/) — 必看 tutorial
- 📄 [FastAPI + LangChain 集成](https://python.langchain.com/docs/langserve/)

### 任务清单
- [ ] 写 `api/main.py`(4h)
- [ ] 实现流式响应(1.5h)
- [ ] 测试 5 条问题(1.5h)

### 时间分配
**7h**

---

## 📚 D19(第 19 天)Streamlit 前端

### 学习目标
- [ ] Streamlit 写聊天界面
- [ ] 显示检索过程(可视化"思考路径")
- [ ] 接入 FastAPI

### 任务清单
- [ ] 写 `ui/app.py`(4h)
- [ ] 接入 FastAPI(2h)
- [ ] 美化 UI(1.5h)

### 时间分配
**7.5h**

---

## 📚 D20(第 20 天)Docker 化 + 部署

### 学习目标
- [ ] 写 Dockerfile + docker-compose
- [ ] 一键启动
- [ ] 写部署文档

### 任务清单
- [ ] 写 Dockerfile(2h)
- [ ] 写 docker-compose.yml(1.5h)
- [ ] 本地测试启动(1h)
- [ ] 写 `DEPLOY.md`(2h)

### 时间分配
**6.5h**

---

## 📚 D21(第 21 天)性能优化 + LangSmith 接入

### 学习目标
- [ ] Embedding 缓存
- [ ] 接入 LangSmith(可选)
- [ ] 性能 profiling

### 任务清单
- [ ] 写 embedding 缓存层(2.5h)
- [ ] 接入 LangSmith(2h)
- [ ] 跑性能测试(2h)

### 时间分配
**6.5h**

---

## 📚 D22(第 22 天)Stage 5 完结

### 任务清单
- [ ] 写 Stage 5 总结(1.5h)
- [ ] 录 5 分钟 demo 视频(2h)
- [ ] 整理项目文档(2h)
- [ ] Git tag:`v4.0-production`

### 时间分配
**5.5h**

---

# Stage 6: 前沿 + 简历整合(D23-D30)

> **本阶段目标**:了解前沿 + 把所有项目整理成可投递版本。

## 📚 D23-D24(周末)GraphRAG 入门

### 学习目标
- [ ] 理解 GraphRAG vs Vector RAG
- [ ] 跑通微软 GraphRAG demo
- [ ] 理解适用场景

### 资料清单
- 📄 [GraphRAG 论文(2024)](https://arxiv.org/abs/2404.16130) — 微软
- 🔧 [GraphRAG 官方 GitHub](https://github.com/microsoft/graphrag) — 必跑
- 📄 [LightRAG 论文](https://arxiv.org/abs/2410.05779) — 轻量版

### 任务清单
- [ ] 读 GraphRAG 论文(2h)
- [ ] 跑通官方 demo(3h)
- [ ] 与自己 RAG 对比(2h)
- [ ] 笔记(1h)

### 时间分配
**8h**

## 📚 D25(第 25 天)Long-Context vs RAG 讨论

### 学习目标
- [ ] 了解 2024-2025 关于"长上下文能否取代 RAG"的讨论
- [ ] 了解 Gemini 1M / Claude 200K 时代的 RAG 价值

### 资料清单
- 📄 [Long Context RAG Survey(2024)](https://arxiv.org/abs/2407.16833) — 综述
- 🎥 [Various YouTube discussions on this topic](https://www.youtube.com/results?search_query=long+context+vs+rag)

### 任务清单
- [ ] 读综述(2h)
- [ ] 写自己的观点笔记(1.5h)
- [ ] 复盘自己 RAG 项目是否还有价值(1.5h)

### 时间分配
**5h**

## 📚 D26(第 26 天)HyDE + RAG-Fusion 拓展

### 学习目标
- [ ] 实现 HyDE 假想文档检索
- [ ] 实现 RAG-Fusion 多查询融合

### 任务清单
- [ ] 写 `hyde.py`(2h)
- [ ] 写 `rag_fusion.py`(2h)
- [ ] 跑 A/B 测试(1.5h)

### 时间分配
**5.5h**

## 📚 D27-D28(周末)简历整合 + 多版本生成

### 学习目标
- [ ] 把 30 天学习内容整理成 4-5 个简历项目
- [ ] 生成 2-3 个针对不同公司(JD 关键词不同)的简历版本
- [ ] 准备面试 STAR 故事

### 任务清单
- [ ] 整理简历项目(3h)
- [ ] 生成 2 个公司版本(腾讯 / 字节 vs 米哈游 / 吉比特)(2h)
- [ ] 准备 5 个 STAR 故事(3h)
- [ ] GitHub 仓库美化(README + 截图)(2h)

### 时间分配
**10h**

## 📚 D29-D30(周末)模拟面试 + 投递准备

### 学习目标
- [ ] 跑 3 轮模拟面试(我当面试官)
- [ ] 整理高频问题答案
- [ ] 投递首批公司

### 任务清单
- [ ] 模拟面试(2h × 3 = 6h)
- [ ] 整理面试题库(2h)
- [ ] 投递 + 跟进(2h)

### 时间分配
**10h**

---

# 📚 关键论文清单(必读 / 选读)

## 必读 ⭐⭐⭐

| 论文 | 年份 | 必读章节 | 用时 |
|---|---|---|---|
| [Lewis RAG 原论文](https://arxiv.org/abs/2005.11401) | 2020 | Abstract + Introduction | 30min |
| [Sentence-BERT](https://arxiv.org/abs/1908.10084) | 2019 | Abstract + 模型图 | 20min |
| [BGE](https://arxiv.org/abs/2309.07597) | 2023 | 架构 + 训练 | 45min |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | 2023 | 全文 | 2h |
| [CRAG](https://arxiv.org/abs/2401.15884) | 2024 | Abstract + Method | 1h |
| [Adaptive-RAG](https://arxiv.org/abs/2403.14403) | 2024 | Abstract + Method | 1h |
| [RAGAS](https://arxiv.org/abs/2309.15217) | 2023 | Abstract + 4 指标定义 | 1h |
| [HyDE](https://arxiv.org/abs/2212.10496) | 2022 | Abstract + 1 个例子 | 20min |

## 选读 ⭐⭐

| 论文 | 年份 | 用途 |
|---|---|---|
| [GraphRAG](https://arxiv.org/abs/2404.16130) | 2024 | 知识图谱融合 |
| [FLARE](https://arxiv.org/abs/2305.06983) | 2023 | 主动式检索 |
| [Long Context vs RAG](https://arxiv.org/abs/2407.16833) | 2024 | 前沿思考 |
| [RRF 论文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | 2021 | 融合公式 |
| [Step-back Prompting](https://arxiv.org/abs/2310.06117) | 2023 | 改写策略 |
| [LightRAG](https://arxiv.org/abs/2410.05779) | 2024 | 轻量 GraphRAG |

---

# 🛠️ 工具与资源索引

## Python 包清单

```bash
# 核心
pip install langchain langchain-community langchain-huggingface
pip install langchain-chroma langchain-ollama langchain-openai
pip install langgraph
pip install chromadb
pip install sentence-transformers
pip install FlagEmbedding

# 评估
pip install ragas datasets

# 工程化
pip install fastapi uvicorn streamlit
pip install pydantic pydantic-settings

# 可视化
pip install matplotlib seaborn
pip install langsmith

# 工具
pip install rank_bm25 jieba
pip install openpyxl
```

## 关键文档链接

| 工具 | 链接 | 优先级 |
|---|---|---|
| LangChain 官方 | https://python.langchain.com/ | ⭐⭐⭐ |
| LangGraph 官方 | https://langchain-ai.github.io/langgraph/ | ⭐⭐⭐ |
| LangGraph Agentic RAG 教程 | https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/ | ⭐⭐⭐ |
| RAGAS 文档 | https://docs.ragas.io/ | ⭐⭐⭐ |
| LangChain 高级 RAG | https://python.langchain.com/docs/tutorials/rag/ | ⭐⭐⭐ |
| 微软 GraphRAG | https://github.com/microsoft/graphrag | ⭐⭐ |
| HuggingFace BGE | https://huggingface.co/BAAI/bge-reranker-base | ⭐⭐⭐ |
| FastAPI 官方 | https://fastapi.tiangolo.com/zh/ | ⭐⭐ |

## 推荐视频

| 视频 | 时长 | 必看? |
|---|---|---|
| [吴恩达 Agentic RAG 短课](https://www.deeplearning.ai/short-courses/agentic-rag/) | 1h | ⭐⭐⭐ |
| [LangGraph 官方教程系列](https://www.youtube.com/@LangChain) | 选看 | ⭐⭐ |
| [Advanced RAG 综述视频](https://www.youtube.com/results?search_query=advanced+rag+techniques) | 1h | ⭐⭐ |

---

# 🎯 30 天后简历可写项目

> 这一节是 **30 天后**你简历上能写的项目(基于真实代码)。

## 项目 1:Advanced RAG 知识库问答系统

> 基于 LangChain + Chroma + BGE + Ollama 搭建的 Advanced RAG 系统,实现混合检索(BGE-Reranker 精排)+ Query 改写(MultiQuery + HyDE),用 RAGAS 4 指标评估,准确率从 65% 提升到 89%。

**核心数据**:
- 30+ 条测试集
- 4 个 RAGAS 指标 baseline vs 改造后
- 完整 A/B 对比图

## 项目 2:Agentic RAG 智能检索系统

> 基于 LangGraph StateGraph 实现多路径决策的 RAG 系统,集成 Router(LLM 决策)、Grader(检索质量评分)、Rewriter(查询改写重查)、Reflector(答案自评幻觉)四大能力,支持 Self-RAG / CRAG / Adaptive-RAG 范式。

**核心数据**:
- 8 个 LangGraph 节点
- 30+ 条评估数据
- 完整流程可视化

## 项目 3:RAG 评估 Pipeline(RAGAS 工程化)

> 用 RAGAS 框架构建 RAG 评估 pipeline,支持 context_precision / context_recall / faithfulness / answer_relevancy 4 指标,作为 RAG 项目的"质量门禁"。

## 项目 4:Web 部署版本(FastAPI + Streamlit)

> 将 RAG 系统 FastAPI 化 + Streamlit Web UI,Docker 一键部署,支持流式响应,LangSmith 全链路追踪。

## 项目 5(可选):GraphRAG 探索

> 基于微软 GraphRAG 实现知识图谱 + 向量混合检索,适合全局性问题。

---

# ⚠️ 学习关键提醒

## 整体节奏

- **每天 4-6h**,周末 8h,**别贪多**:一口气学 12h 第二天就废
- **每天必有产出**:`Git commit` / `可运行代码` / `笔记 md`
- **遇到坑就记**:不是所有问题都要 1 天解决,卡 2 天很正常

## 常见踩坑

1. **Ollama 未启动**:`ConnectionRefusedError localhost:11434` → 跑 `ollama serve`
2. **HF Token 缺失**:BGE 等模型下载要 token,`export HF_TOKEN=xxx`
3. **RAGAS 安装失败**:`pip install ragas` 要 `datasets` 配套
4. **LangGraph 导入路径**:0.2+ 路径经常变,以官方文档为准
5. **显存不够**:BGE-Reranker base 模型约 1.1GB,本地跑需要 4GB+ 显存或用 CPU(慢)

## 评估准则

- 每天结束问自己:**今天产出了什么可运行的东西?**
- 每周结束问自己:**这周学的内容能在简历上写 1 行吗?**
- 30 天结束问自己:**这 5 个项目每个能讲 5 分钟吗?**

---

# 📅 时间表(2026 年 6-7 月)

| 周次 | 日期 | 阶段 | 周末计划 |
|---|---|---|---|
| W1 | 6/7 - 6/13 | Stage 1(D1-D6)Advanced RAG + 评估基础 | 周末加量 |
| W2 | 6/14 - 6/20 | Stage 2-3(D7-D14)评估 + 切块 + Agentic 启动 | GraphRAG 启动 |
| W3 | 6/21 - 6/27 | Stage 4 中后期(D15-D22)Agentic + 工程化 | 完整 demo |
| W4 | 6/28 - 7/4 | Stage 5-6 收尾 + 简历整合 | 投递准备 |
| W5 | 7/5 - 7/6 | 模拟面试 + 投递启动 | — |

**关键节点**:
- 6/13:Stage 1-2 完结 → 简历第一个项目素材到手
- 6/20:Stage 3-4 完结 → 简历核心项目素材到手
- 6/27:Stage 4-5 完结 → 可以开始投递实习
- 7/4:全部 5 个项目完结 → 全力投递秋招提前批

---

# 🚀 启动方式

**今天就开始 D1**:

```bash
# 1. 进入项目目录
cd D:\Pythoncode\RAG

# 2. 创建笔记目录
mkdir notes
mkdir reports
mkdir data

# 3. 安装基础包
pip install langchain langchain-community langchain-huggingface langchain-chroma
pip install chromadb sentence-transformers rank_bm25

# 4. 创建 D1 笔记文件
# echo "# D1 - 混合检索" > notes/D1_hybrid_search.md

# 5. 开始读 Advanced RAG 综述
# 打开 https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8f90d
```

---

**任何问题随时找我**。每个 Day 结束如果你想 review 一下,我可以直接读你的代码 + 笔记,给你具体的反馈。

**Day 1 任务清单**:
- [ ] 读 Advanced RAG 综述
- [ ] 读 RRF 公式部分
- [ ] 实现 `hybrid_retriever.py`
- [ ] A/B 对比 5 条问题
- [ ] Git commit

**准备好了就开始**。我等你 D1 结束过来 review。
