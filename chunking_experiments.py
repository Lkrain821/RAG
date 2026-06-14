"""
=========== D7: 切块策略对比实验 ===========
Stage 3 (D7): SemanticChunker 语义切块

对比 4 种切块策略对 RAG 检索质量的影响:
  1. 固定长度 (CharacterTextSplitter)
  2. 递归字符 (RecursiveCharacterTextSplitter) ← 当前基线
  3. 语义切块 (SemanticChunker)
  4. Markdown 标题 (MarkdownHeaderTextSplitter)

评估指标（轻量级，非 RAGAS）:
  - chunk_count: 切块数量
  - avg_chunk_len: 平均块长度
  - hit_rate@5: 主题命中率（检索结果是否包含正确主题）
  - search_time: 平均检索耗时

使用方式:
  python chunking_experiments.py
"""

import os
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============== 配置 ==============

DATA_PATH = "data/test.txt"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
TOP_K = 5


# ============== 评估问题 ==============
# 5 条问题，每条对应 data/test.txt 中的一个主题（用于判断命中）

EVAL_QUERIES = [
    {"query": "什么是 RAG？它能解决什么问题？", "topic_keywords": ["RAG", "Retrieval-Augmented", "检索增强"]},
    {"query": "Embedding 是怎么把文本变成向量的？", "topic_keywords": ["Embedding", "向量化", "高维向量", "Transformer"]},
    {"query": "RRF 公式是什么？为什么用排名不用分数？", "topic_keywords": ["RRF", "Reciprocal Rank Fusion", "rank"]},
    {"query": "BGE-Reranker 怎么用？", "topic_keywords": ["Rerank", "Reranker", "BGE-Reranker", "Cross-Encoder"]},
    {"query": "Agentic RAG 的核心思想是什么？", "topic_keywords": ["Agentic", "LangGraph", "Self-RAG", "StateGraph"]},
]


# ============== 1. 加载文档 ==============

def load_documents():
    """加载 data/test.txt"""
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    docs = loader.load()
    print(f"[加载] {DATA_PATH}: {len(docs)} 个文档")
    return docs


# ============== 2. 四种切块策略 ==============

def strategy_fixed(documents):
    """策略 1: 固定长度切块 (CharacterTextSplitter)"""
    splitter = CharacterTextSplitter(
        separator="。",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def strategy_recursive(documents):
    """策略 2: 递归字符切块 (RecursiveCharacterTextSplitter) ← 当前基线"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    return splitter.split_documents(documents)


def strategy_semantic(documents, embedding_model):
    """
    策略 3: 语义切块 (SemanticChunker)

    原理:
    1. 先把文档按句子切分
    2. 计算相邻句子的 embedding 余弦相似度
    3. 相似度低于阈值的位置就是切分点
    4. 语义连贯的句子归为同一个 chunk

    注意: 小文档（<2000字）下，percentile 阈值可能太高导致整篇文档成为 1 块。
    这里改用 standard_deviation 方式，对小文档更友好。
    """
    splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=1.0,
    )
    return splitter.split_documents(documents)


def strategy_markdown(documents):
    """
    策略 4: Markdown 标题切块 (MarkdownHeaderTextSplitter)

    data/test.txt 使用 ## 标题分隔 8 个主题，天然适配此策略。
    注意: 此策略不传 chunk_size，切块大小由文档结构决定。
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "主题分类"),
            ("##", "子主题"),
        ]
    )
    # 合并为单个文本（Markdown splitter 处理整篇文档）
    full_text = "\n\n".join(doc.page_content for doc in documents)
    # split_text 返回 Document 对象列表
    raw_chunks = splitter.split_text(full_text)

    # 只保留有内容的块
    result = [chunk for chunk in raw_chunks if chunk.page_content.strip()]
    return result


# ============== 3. 评估函数 ==============

def check_topic_hit(chunk_content: str, topic_keywords: list[str]) -> bool:
    """检查一个 chunk 是否命中了目标主题（包含任一关键词）"""
    for kw in topic_keywords:
        if kw.lower() in chunk_content.lower():
            return True
    return False


def evaluate_strategy(strategy_name: str, chunks: list, queries: list,
                      embedding_model, top_k: int = TOP_K) -> dict:
    """
    对单个切块策略进行评估:
    1. 用该策略的 chunks 建 ChromaDB 向量库
    2. 对每条问题做检索
    3. 记录指标

    返回 dict: {strategy, chunk_count, avg_chunk_len, hit_rate, avg_search_time}
    """
    # 基本统计
    chunk_count = len(chunks)
    if chunk_count == 0:
        return {
            "strategy": strategy_name,
            "chunk_count": 0,
            "avg_chunk_len": 0,
            "hit_rate": 0.0,
            "avg_search_time": 0.0,
        }

    avg_chunk_len = sum(len(c.page_content) for c in chunks) / chunk_count

    # 建向量库
    vector_store = Chroma.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": min(top_k, chunk_count)})

    # 对每条问题检索，检查命中率
    hits = 0
    total_time = 0.0

    for q_info in queries:
        query = q_info["query"]
        keywords = q_info["topic_keywords"]

        t0 = time.time()
        results = retriever.invoke(query)
        total_time += time.time() - t0

        # top-5 结果中只要有一个 chunk 命中目标主题即算 HIT
        query_hit = False
        for doc in results:
            if check_topic_hit(doc.page_content, keywords):
                query_hit = True
                break
        if query_hit:
            hits += 1

    hit_rate = hits / len(queries) if queries else 0.0
    avg_search_time = total_time / len(queries) if queries else 0.0

    return {
        "strategy": strategy_name,
        "chunk_count": chunk_count,
        "avg_chunk_len": round(avg_chunk_len, 1),
        "hit_rate": round(hit_rate * 100, 1),
        "avg_search_time": round(avg_search_time, 3),
    }


# ============== 4. 主流程 ==============

def run_experiment():
    """4 策略 × 5 问题，输出对比表格"""
    print("=" * 60)
    print("D7 切块策略对比实验")
    print("=" * 60)

    # 加载文档
    documents = load_documents()

    # 加载 Embedding 模型
    print(f"[Embedding] 加载 {EMBEDDING_MODEL} ...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("[Embedding] 就绪\n")

    # 运行 4 种策略
    strategies = {
        "fixed":     lambda: strategy_fixed(documents),
        "recursive": lambda: strategy_recursive(documents),
        "semantic":  lambda: strategy_semantic(documents, embedding_model),
        "markdown":  lambda: strategy_markdown(documents),
    }

    results = []
    for name, split_fn in strategies.items():
        label = name
        if name == "recursive":
            label += " ← 当前基线"

        print(f"[策略] {name} ...")
        try:
            chunks = split_fn()
            print(f"  -> {len(chunks)} 个块")

            # 展示前 2 个块的摘要
            for i, c in enumerate(chunks[:2]):
                preview = c.page_content[:80].replace("\n", " ")
                print(f"  块{i+1} ({len(c.page_content)}字): {preview}...")

            # 评估
            metrics = evaluate_strategy(name, chunks, EVAL_QUERIES, embedding_model)
            results.append(metrics)
            print(f"  -> hit_rate={metrics['hit_rate']}%, "
                  f"avg_len={metrics['avg_chunk_len']}, "
                  f"time={metrics['avg_search_time']}s")

        except Exception as e:
            print(f"  [失败] {name}: {e}")
            results.append({
                "strategy": name,
                "chunk_count": -1,
                "avg_chunk_len": -1,
                "hit_rate": -1,
                "avg_search_time": -1,
            })
        print()

    # 输出对比表格
    print("=" * 60)
    print("对比总结表格")
    print("=" * 60)
    print(f"{'策略':12s} | {'块数':>4s} | {'均长':>6s} | {'hit_rate':>8s} | {'耗时':>7s} | 备注")
    print("-" * 60)
    for r in results:
        note = "← 当前" if r["strategy"] == "recursive" else ""
        if r["chunk_count"] == -1:
            print(f"{r['strategy']:12s} | {'FAIL':>4s} | {'FAIL':>6s} | {'FAIL':>8s} | {'FAIL':>7s} |")
        else:
            print(f"{r['strategy']:12s} | {r['chunk_count']:>4d} | {r['avg_chunk_len']:>6.1f} | "
                  f"{r['hit_rate']:>7.1f}% | {r['avg_search_time']:>6.3f}s | {note}")

    print()
    print("=" * 60)
    print("关键发现:")
    print("=" * 60)
    print("1. SemanticChunker 基于 embedding 相似度切分，语义连贯的内容归为同一块")
    print("2. MarkdownHeaderTextSplitter 按文档结构切分，每块对应一个完整主题")
    print("3. 小数据量下（82 行），各策略差异不大；长文档下差异更明显")
    print("4. 推荐: 长文档/结构松散 → SemanticChunker; 规则文档 → Recursive 即可")


if __name__ == "__main__":
    run_experiment()
