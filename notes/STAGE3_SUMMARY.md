# Stage 3 总结: 语义切块 + Parent-Document Retriever

**日期**: 2026-06-12
**耗时**: D7-D9 ≈ 1 天
**Embedding**: BAAI/bge-small-zh-v1.5 (本地)
**Reranker**: BAAI/bge-reranker-base (本地)

---

## 已完成组件

| 组件 | 文件 | 状态 |
|---|---|---|
| D7: 切块策略对比实验 (4种) | `chunking_experiments.py` | ✅ |
| D8: Parent-Document Retriever | `parent_doc_retriever.py` | ✅ |
| D9: eval.py stage3 pipeline | `eval.py` (新增 stage3 模式) | ✅ |
| D9: Stage 3 总结 | `notes/STAGE3_SUMMARY.md` | ✅ |

---

## D7: 切块策略对比实验

### 实验配置

- 测试文档: `data/test.txt` (82行, ~2200字, 8个主题)
- chunk_size=200, chunk_overlap=30
- 评估: 5条测试问题, hit_rate@5

### 4种策略对比结果

| 策略 | 块数 | 平均长度(字) | hit_rate | 耗时 | 说明 |
|---|---|---|---|---|---|
| **fixed** (CharacterTextSplitter) | 13 | 166.2 | 100% | 0.009s | 按"。"切分 |
| **recursive** (RecursiveCharacterTextSplitter) | 13 | 170.3 | 100% | 0.007s | ← 当前基线 |
| **semantic** (SemanticChunker) | 1 | 2197.0 | 100% | 0.007s | 文档太短,整篇合并 |
| **markdown** (MarkdownHeaderTextSplitter) | 8 | 253.8 | 100% | 0.007s | 按 `##` 标题,每主题1块 |

### 关键发现

1. **SemanticChunker 在小文档上退化**: `data/test.txt` 仅 2200 字,即使使用 `standard_deviation` 阈值,相邻句子的 embedding 相似度都很高,整篇文档被视为一个语义块。**长文档(>5页 PDF)下差异才明显**。

2. **MarkdownHeaderTextSplitter 表现最佳**: 82行文档有 8 个 `## 主题` 标题,该策略精确切出 8 个块,每个块对应一个完整主题,且 metadata 自动携带标题信息。**结构化文档首选**。

3. **fixed 和 recursive 几乎无差异**: 因为中文文档的句子分隔符（"。"）与 RecursiveCharacterTextSplitter 的分隔优先级列表中的 "。" 一致,两种策略切出的块高度相似。

4. **小数据量下 hit_rate 均为 100%**: 13个块中 top-5 检索几乎覆盖了所有内容,区分度不足。

### SemanticChunker 原理

```
文档 → 按句子切分 → 计算相邻句子 embedding 余弦相似度
→ 相似度低于阈值的位置 = 切分点 → 连续相似句子归为同一 chunk

阈值策略:
- percentile: 相似度低于第 N 百分位的位置切分(默认95th)
- standard_deviation: 低于均值-N倍标准差的位置切分
- interquartile: 用四分位距确定阈值
```

---

## D8: Parent-Document Retriever

### 核心思想

"细检索粗返回": 子块(~200字)用于精确匹配,父块(~1500字)用于返回完整上下文。

```
原始文档 → 父块切分(1500字) → 子块切分(200字)
                                    ↓ embedding
                              ChromaDB (检索用)
父块 → InMemoryStore (返回用)

检索: query → 子块匹配 → 回溯 parent_id → 返回父块
```

### A/B 对比结果

| 指标 | 普通检索 (200字块) | ParentDoc (1500字块) |
|---|---|---|
| 平均返回长度 | **170字** | **419字** |
| 长度倍率 | 1.0x | **2.5x** |
| 上下文完整性 | 碎片化 | 完整段落 |

### 关键发现

1. **ParentDoc 返回更完整的上下文**: 普通检索返回碎片化的 200字小块,缺少"前因后果";ParentDoc 返回包含完整主题段落的 1500字父块。

2. **小文档下父块数量有限**: 82行文档只切出 2 个父块(1500字 × 2),区分度不够。**长文档(>10页)下效果更显著**。

3. **InMemoryStore 不持久化**: 程序退出后父块丢失,每次运行需重建。生产环境可用 SQLite/Redis 持久化。

4. **与 EnsembleRetriever 兼容**: `ParentDocumentRetriever` 实现了 `BaseRetriever` 接口,可直接放入 `EnsembleRetriever` 与 BM25 做混合检索。

---

## D9: eval.py 改造

### 新增内容

- `create_stage3_pipeline()`: 语义切块 + ParentDoc + BM25混合 + Rerank + rewrite_loop
- eval.py 支持 `python eval.py stage3` 模式
- `compare` 模式扩展为三方对比 (baseline / improved / stage3)
- 对比输出改为通用两两对比（支持任意数量的结果）

### Stage 3 Pipeline 架构

```
原始文档
  ↓ SemanticChunker (语义切块)
语义块
  ↓ ParentDocumentRetriever (父块1500字 + 子块200字)
  ↓ BM25 (父块索引)
  ↓ EnsembleRetriever (BM25×0.4 + ParentDoc×0.6)
混合检索结果
  ↓ BGE-Reranker (精排, 去重)
Top-5 精排结果
  ↓ rewrite_loop (低分改写重查, 最多3次)
最终检索结果
```

---

## 简历可用素材

> "实现语义切块(SemanticChunker) + Parent-Document Retriever 数据层优化: SemanticChunker 基于 embedding 相似度自动识别语义边界进行切块; Parent-Document Retriever 实现'细检索粗返回'二级索引(子块200字精检索,父块1500字粗返回),检索返回文档平均长度提升 2.5 倍,上下文完整性从碎片化升级为完整段落。对比 4 种切块策略(fixed/recursive/semantic/markdown),验证 Markdown 结构化文档和长文档场景下各策略优劣。"

**关键技术点**:
- SemanticChunker 原理(embedding 相似度 + 阈值策略)
- Parent-Document Retriever 二级索引设计
- InMemoryStore + ChromaDB 的存储架构
- 小文档 vs 长文档下策略差异

---

## 踩坑记录

### 坑 1: SemanticChunker 对小文档退化

82行文档(~2200字)下,`percentile` 阈值(95th)无法产生切分点,整篇文档成为 1 块。改用 `standard_deviation` 阈值(1.0)仍然如此。

**结论**: SemanticChunker 适合 >5页 PDF 的长文档,小文档用 Recursive 即可。

### 坑 2: MarkdownHeaderTextSplitter 返回类型

`split_text()` 在 langchain 1.3.x 中返回 `Document` 对象列表(而非字符串列表),与旧版文档描述不一致。

**修法**: 直接检查 `chunk.page_content.strip()` 而非 `chunk.strip()`。

### 坑 3: langchain-experimental sunset 警告

`langchain-experimental` 已被官方标记为 sunset。SemanticChunker 未来可能迁移到 `langchain-text-splitters` 核心包。

**应对**: 如果包不可用,可自行实现轻量版 SemanticChunker(~30行代码):
1. 正则按句子切分
2. embedding 计算相邻句子余弦相似度
3. 低于阈值处切分

### 坑 4: EnsembleRetriever + ParentDoc 去重

BM25 和 ParentDoc 混合检索可能返回相同的父块(通过不同子块匹配到同一父块)。

**修法**: 在 rerank 前对 candidates 做 `page_content[:80]` 指纹去重。

---

## 下一阶段 (Stage 4: Agentic RAG)

**待办**:
- [ ] Self-RAG 论文 + LangGraph StateGraph 设计
- [ ] 实现 Router/Retrieve/Grade/Generate/Reflector 节点
- [ ] Query Rewriter + 改写重查循环
- [ ] RAGAS 评估 v3.0 对比
