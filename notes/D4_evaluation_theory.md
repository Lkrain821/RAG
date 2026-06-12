# D4 - RAG 评估理论基础 学习笔记

**日期**: 2026-06-10
**目标**: 理解 RAGAS 4 个核心指标的含义与评估方法论

---

## 1. 为什么要做评估？

> 没评估你不知道改动有没有用——这是 Stage 2 放在 Stage 1 之后的原因。

- 面试时拿出 4 个评估指标 + 数字，从"做过 RAG"升级到"懂 RAG"
- 评估是优化的前置条件：先有 baseline，每次改动跑一遍，用数据驱动决策
- 避免"感觉变好了"的主观判断

---

## 2. RAGAS 4 个核心指标

| 指标 | 衡量什么 | 输入依赖 | LLM Judge? |
|---|---|---|---|
| **context_precision** | 检索结果中真正相关内容的占比 | question + contexts + reference | ✅ |
| **context_recall** | 应该召回的有没有召回来 | question + contexts + reference | ✅ |
| **faithfulness** | 答案是否完全基于检索内容（防幻觉） | question + contexts + answer | ✅ |
| **answer_relevancy** | 答案与问题的相关度 | question + answer + embeddings | ✅ |

### context_precision (上下文精确度)

- 公式本质：检索到的 N 个文档中，有多少是真正与 ground_truth 相关的
- 高 precision = 检索回来的东西大部分有用
- 低 precision = 检索混入了无关文档，浪费了 LLM 的上下文窗口

### context_recall (上下文召回率)

- 公式本质：ground_truth 中需要的信息，检索系统覆盖了多少
- 高 recall = 该找的都找到了
- 低 recall = 漏掉了关键信息，LLM 无法给出完整答案

### faithfulness (忠实度)

- **防幻觉的关键指标**
- 把答案拆成若干声明（claims），逐一检查是否能在检索上下文中找到依据
- 高 faithfulness = 答案没有编造内容
- 低 faithfulness = 答案包含了文档中没有的信息（幻觉）

### answer_relevancy (答案相关度)

- 衡量答案是否切题
- 用 LLM 基于答案反向生成多个问题，计算生成问题与原问题的语义相似度
- 高 relevancy = 答案紧扣问题
- 低 relevancy = 答案跑题或冗余

---

## 3. LLM-as-Judge 范式

RAGAS 使用"LLM 做裁判"模式评估 RAG 输出：

```
Question + Contexts + Answer → Judge LLM → Score
```

**关键认知**：
- Judge LLM 必须"足够强" —— GPT-4 / DeepSeek-V3 级别
- 本地小模型（Ollama qwen3:4b）做 judge **偏高且不稳定**，评估结果不可信
- 成本：30 条 × 4 指标 × ~3 次 LLM 调用/指标 ≈ 360 次 API 调用

---

## 4. 评估实践中的注意事项

### 4.1 数据集设计

- 覆盖所有主题（本文档 8 个主题，34 条问题）
- 问题类型多样：定义类、比较类、How-to 类、深层理解类
- ground_truth 必须基于文档内容精确撰写，不能"凭感觉"

### 4.2 小数据集的局限性

本文档仅 13 个 chunk，存在天然瓶颈：
- context_recall 几乎必然为 1.0（文档太小，k=5 几乎全覆盖）
- faithfulness 容易满分（答案短、上下文少）
- context_precision 和 answer_relevancy 是更有区分度的指标

### 4.3 DeepSeek API 的兼容性问题

- DeepSeek API 不支持 `n>1`（多候选生成），只返回 1 个 generation
- 影响 answer_relevancy 的计算精度（原设计需生成 3 个反向问题取平均相似度）
- **结论**：answer_relevancy 分数可能偏低，仅作参考

---

## 5. 关键公式与概念速查

| 概念 | 一句话 |
|---|---|
| RAGAS | RAG 评估的事实标准框架 |
| context_precision | `|相关文档 ∩ 检索文档| / |检索文档|` |
| context_recall | `|检索到的相关文档| / |所有相关文档|` |
| faithfulness | 答案 claims 中能在上下文中验证的比例 |
| answer_relevancy | 反向生成问题的语义相似度均值 |
| LLM-as-Judge | 用强 LLM 替代人工评估的范式 |
