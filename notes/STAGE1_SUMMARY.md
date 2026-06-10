# Stage 1 总结: Advanced RAG 三大件

**日期**: 2026-06-10
**耗时**: D1(6/7) + D2(6/8) + D3(6/10) ≈ 3 天

---

## 已完成组件

| 组件 | 文件 | 状态 |
|---|---|---|
| D1: BM25 + 向量 + RRF 混合检索 | `hybrid_retriever.py` | ✅ |
| D2: BGE-Reranker Cross-Encoder 精排 | `rerank_hybrid.py` | ✅ |
| D2: 2×2 Ablation Study | `ablation_hybrid_only.py` | ✅ |
| D3: MultiQuery 改写 | `query_rewriter.py` | ✅ |
| D3: HyDE 假想文档检索 | `query_rewriter.py` | ✅ |
| D3: 重查循环 (rewrite_loop) | `query_rewriter.py` | ✅ |

---

## D3 关键发现

### 测试环境
- 13 个文档块 (chunk_size=200), 8 个主题
- 5 条测试问题, 涵盖模糊描述/通用提问/多部分/英文术语/间接问题
- BGE-Reranker 分数阈值: 0.0 (高于 0 为通过)

### Ablation 结果

| 策略 | Score Pass | Topic Correct | 延迟 |
|---|---|---|---|
| **baseline** (D2 流水线) | 4/5 | 3/5 | ~2s/条 |
| **multi_query** (LLM 拆多视角) | 4/5 | 3/5 | +30s/条 |
| **hyde** (假想答案检索) | 4/5 | 3/5 | +40s/条 |
| **rewrite_loop** (重查循环) | **5/5** | **3/5** | +30s/条 (仅 Q5) |

### 核心发现

1. **rewrite_loop 是唯一有效策略**:
   - Q5 "幻觉问题怎么解决?" baseline score=-1.99 (低于阈值)
   - rewrite_loop 改写后 score=+4.69, 提升 6.68 分
   - 命中同一正确文档 (chunk[10]: faithfulness), 但改写使查询更贴合文档语言

2. **multi_query 和 hyde 在小数据集上无效**:
   - 13 个块太少, BM25+向量+k=20 已近乎全量召回
   - multi_query 拆多视角未超出已有召回范围
   - hyde 生成的假想答案与真实文档风格差异大, 未提升匹配度
   - **两者都增加 30-40s 延迟, 零收益**

3. **Q3/Q4 所有策略都错**:
   - Q3 "向量数据库选型?" → 命中 BGE-small-zh 段落 (主题3)，非向量数据库段落 (主题4)
   - Q4 "Cross-Encoder 怎么做打分?" → 命中 RRF 公式段落 (主题5)，非 BGE-Reranker 段落 (主题6)
   - **根因**: chunk 边界问题导致混合检索的语义定位偏差
   - 这是数据层(切块/主题粒度)的问题, 不是改写能解决的

### 结论

- **rewrite_loop 有价值**: 对低分检索能有效提升置信度, 值得保留
- **multi_query 和 hyde 在此场景暂不投入**: 数据量大了以后 (50+ chunks) 可能有收益
- **当前瓶颈在数据层**: 切块策略和文档质量是更大的杠杆

---

## Stage 1 完整流水线

D3 最终架构: `query → [rewrite_loop] → hybrid_search(k=20) → rerank(top=5) → score_check → [retry if needed]`

```python
# 使用方式
from query_rewriter import search

# 默认策略 (D2 baseline)
results, scores = search(query, strategy="baseline")

# 带重查循环 (D3 推荐)
results, scores, loops = search(query, strategy="rewrite_loop")
```

---

## 简历可用素材

> "实现 Advanced RAG 三大件: BM25+Dense 混合检索、BGE-Reranker Cross-Encoder 精排、重查循环 Query 改写。通过 2×2 控制变量实验量化优化效果: 细粒度分块 (+30pp)、Cross-Encoder 精排 (+33pp)、重查改写 (低分检索置信度 +6.68)。"

**关键技术点**:
- RRF 融合 (k=60)
- Cross-Encoder vs Bi-Encoder
- BGE-Reranker-base (中文首选)
- 重查循环: score_threshold + LLM改写 + 重查
- 控制变量实验设计 (Ablation Study)

**生产坑**:
- FlagEmbedding 1.4.0 + transformers 5.x 不兼容
- chunk 太粗时 Rerank 无效果 (需 ≤200)
- MultiQuery/HyDE 在小数据集上加延迟无收益

---

## 下一阶段 (Stage 2: RAGAS 评估)

**待办**:
- [ ] 安装 ragas 包
- [ ] 构建 30+ 条测试集
- [ ] 跑 4 个 RAGAS 指标
- [ ] baseline vs Stage 1 改造后对比报告
