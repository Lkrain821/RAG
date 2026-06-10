# ============== query_rewriter.py ==============
# D3 任务:MultiQuery + HyDE + 重查循环
#
# 在 D2(混合检索 + BGE-Reranker)基础上增加 Query 改写层。
#
# 三种策略对比:
#   baseline: 原版 D2 (hybrid_search -> rerank)
#   multi_query: LLM 生成 3 个不同视角的查询 -> 分别检索 -> 去重融合 -> rerank
#   hyde: LLM 生成假想答案 -> 用假答案检索 -> rerank
#   rewrite_loop: 检索 -> rerank -> score<阈值 -> 改写重查(最多3次)
#
# 使用方式:
#   python query_rewriter.py        # 跑 ablation 对比测试
# ===============================================

import os
import torch
import time
from collections import defaultdict

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import jieba


# ============== 1. 基础设施(D2 复用) ==============

DATA_PATH = "data/test.txt"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "qwen3:4b"
HYBRID_K = 20
TOP_N = 5
RERANK_THRESHOLD = 0.0


def load_documents():
    """加载并分块文档"""
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.split_documents(raw_docs)
    print(f"  文档: {len(docs)} 个块 (chunk_size={CHUNK_SIZE})")
    return docs


def chinese_tokenizer(text: str) -> list[str]:
    return list(jieba.cut(text))


def init_hybrid_retriever(docs):
    """初始化 BM25 + 向量 + RRF 混合检索"""
    bm25 = BM25Retriever.from_documents(docs, preprocess_func=chinese_tokenizer)
    bm25.k = HYBRID_K

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": HYBRID_K})

    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble, vector_store


def init_reranker():
    """加载 BGE-Reranker Cross-Encoder"""
    print(f"  加载 Reranker: {RERANK_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"    -> device={device}")
    return tokenizer, model, device


def init_llm():
    """初始化 Ollama LLM"""
    print(f"  加载 LLM: {LLM_MODEL} ...")
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)
    return llm


# ============== 2. 检索核心函数(D2 复用) ==============

def hybrid_search(query: str, ensemble_retriever) -> list[Document]:
    """BM25 + 向量 + RRF 混合检索"""
    return ensemble_retriever.invoke(query)


def rerank_docs(query: str, candidates: list[Document],
                rerank_tokenizer, rerank_model, device,
                top_n: int = TOP_N) -> tuple[list[Document], list[float]]:
    """BGE-Reranker 精排,返回 (排序后的docs, 对应scores)"""
    if not candidates:
        return [], []

    pairs = [[query, doc.page_content] for doc in candidates]
    with torch.no_grad():
        inputs = rerank_tokenizer(
            pairs, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()

    sorted_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    sorted_docs = [p[0] for p in sorted_pairs]
    sorted_scores = [p[1] for p in sorted_pairs]
    return sorted_docs[:top_n], sorted_scores[:top_n]


def full_pipeline(query: str, ensemble_retriever,
                  rerank_tokenizer, rerank_model, device) -> tuple[list[Document], list[float]]:
    """完整 D2 流水线: hybrid_search -> rerank"""
    candidates = hybrid_search(query, ensemble_retriever)
    return rerank_docs(query, candidates, rerank_tokenizer, rerank_model, device)


# ============== 3. MultiQuery 策略 ==============

MULTI_QUERY_PROMPT = """你是一个检索专家。用户的问题是: {query}

请从3个不同角度改写这个问题,生成3个独立的搜索查询,使得检索系统能找到不同类型的相关内容。
要求:
- 每个改写站在不同视角(如概念定义、实现方法、应用场景、优缺点等)
- 保持原问题的核心意图,但用不同的措辞表达
- 每个查询用自然语言,不要编号
- 每行一个,直接输出查询内容,不要序号或前缀"""


def multi_query_rewrite(query: str, llm) -> list[str]:
    """用 LLM 生成 3 个不同视角的查询"""
    prompt = MULTI_QUERY_PROMPT.format(query=query)
    result = llm.invoke(prompt)
    lines = [line.strip().strip("- ").strip("* ").strip()
             for line in result.strip().split("\n") if line.strip()]
    import re as _re
    cleaned = []
    for line in lines:
        line = _re.sub(r'^\d+[\.\)、]\s*', '', line)
        if line and len(line) > 3:
            cleaned.append(line)
    if not cleaned:
        cleaned = [query]
    queries = [query] + cleaned[:3]
    return queries


def multi_query_search(query: str, ensemble_retriever,
                       rerank_tokenizer, rerank_model, device, llm) -> tuple[list[Document], list[float]]:
    """MultiQuery: 生成多视角查询 -> 分别检索 -> 去重融合 -> rerank"""
    queries = multi_query_rewrite(query, llm)

    all_candidates = []
    seen_content = set()
    for q in queries:
        candidates = hybrid_search(q, ensemble_retriever)
        for doc in candidates:
            fingerprint = doc.page_content[:60]
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                all_candidates.append(doc)

    return rerank_docs(query, all_candidates, rerank_tokenizer, rerank_model, device)


# ============== 4. HyDE 策略 ==============

HYDE_PROMPT = """请根据以下问题,生成一段详细、准确的假设性回答文档。
要求:
- 回答应包含该主题的核心概念、关键细节和专业术语
- 风格如同百科全书条目
- 长度在 150-300 字之间
- 假设你是该领域的专家,直接写出回答内容,不要额外说明

问题: {query}

假设回答:"""


def hyde_search(query: str, ensemble_retriever, vector_store,
                rerank_tokenizer, rerank_model, device, llm) -> tuple[list[Document], list[float]]:
    """HyDE: 生成假想答案 -> 用假答案做语义检索 + BM25 -> rerank"""
    prompt = HYDE_PROMPT.format(query=query)
    hypothetical_answer = llm.invoke(prompt).strip()

    candidates = hybrid_search(hypothetical_answer, ensemble_retriever)

    return rerank_docs(query, candidates, rerank_tokenizer, rerank_model, device)


# ============== 5. 重查循环策略 ==============

REWRITE_PROMPT = """当前检索结果不够理想,无法找到与用户问题相关的信息。
请重新表述下面的问题,使其更精确、包含更多关键词,便于检索系统找到相关内容。

原问题: {query}
检索到的最相关文档片段: {top_content}

优化后的问题(直接输出,不要解释):"""

REWRITE_NO_CONTEXT_PROMPT = """当前检索没有找到任何相关内容。
请从更宽泛或更通用的角度重新表述下面的问题,便于检索系统找到相关内容。

原问题: {query}

优化后的问题(直接输出,不要解释):"""


def rewrite_query(query: str, top_docs: list[Document], llm) -> str:
    """根据检索反馈改写查询"""
    if top_docs:
        top_content = top_docs[0].page_content[:200]
        prompt = REWRITE_PROMPT.format(query=query, top_content=top_content)
    else:
        prompt = REWRITE_NO_CONTEXT_PROMPT.format(query=query)

    rewritten = llm.invoke(prompt).strip()
    rewritten = rewritten.strip('"').strip("'").strip()
    return rewritten


def rewrite_loop_search(query: str, ensemble_retriever,
                        rerank_tokenizer, rerank_model, device, llm,
                        max_loops: int = 3) -> tuple[list[Document], list[float], int]:
    """重查循环: 检索 -> rerank -> score<阈值 -> 改写 -> 重查(最多3次)"""
    current_query = query
    loops = 0
    results, scores = [], []

    for attempt in range(max_loops):
        loops += 1
        candidates = hybrid_search(current_query, ensemble_retriever)
        results, scores = rerank_docs(current_query, candidates,
                                      rerank_tokenizer, rerank_model, device,
                                      top_n=TOP_N)

        if not results:
            current_query = rewrite_query(current_query, [], llm)
            continue

        if scores[0] >= RERANK_THRESHOLD:
            return results, scores, loops

        current_query = rewrite_query(current_query, results, llm)

    if not results:
        candidates = hybrid_search(current_query, ensemble_retriever)
        results, scores = rerank_docs(current_query, candidates,
                                      rerank_tokenizer, rerank_model, device,
                                      top_n=TOP_N)

    return results, scores, loops


# ============== 6. Ablation 对比测试 ==============

TEST_QUERIES = [
    "那个检索增强技术是啥?",                 # 模糊描述
    "怎么让搜索结果更准?",                   # 通用提问
    "向量数据库有哪些?选型要考虑什么?",       # 多部分问题
    "Cross-Encoder 怎么做打分?",              # 英文术语
    "幻觉问题怎么解决?",                     # 间接问题
]


def run_ablation():
    """跑 5 条问题,对比 baseline / multi_query / hyde / rewrite_loop"""
    print("=" * 60)
    print("D3 Ablation: Query 改写策略对比")
    print("=" * 60)

    print("\n[初始化] ...")
    docs = load_documents()
    ensemble, vector_store = init_hybrid_retriever(docs)
    rerank_tokenizer, rerank_model, device = init_reranker()
    llm = init_llm()

    strategies = {
        "baseline":      lambda q: full_pipeline(q, ensemble, rerank_tokenizer, rerank_model, device),
        "multi_query":   lambda q: multi_query_search(q, ensemble, rerank_tokenizer, rerank_model, device, llm),
        "hyde":          lambda q: hyde_search(q, ensemble, vector_store, rerank_tokenizer, rerank_model, device, llm),
        "rewrite_loop":  lambda q: rewrite_loop_search(q, ensemble, rerank_tokenizer, rerank_model, device, llm),
    }

    print(f"\n{'=' * 60}")
    print(f"测试 {len(TEST_QUERIES)} 条问题 x 4 种策略")
    print(f"{'=' * 60}")

    for q_idx, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─' * 60}")
        print(f"Q{q_idx}: {query}")
        print(f"{'─' * 60}")

        for strategy_name, strategy_fn in strategies.items():
            t0 = time.time()

            if strategy_name == "rewrite_loop":
                results, scores, loops = strategy_fn(query)
                extra_info = f" [loops={loops}]"
            else:
                results, scores = strategy_fn(query)
                extra_info = ""

            elapsed = time.time() - t0

            if results and scores:
                first_line = results[0].page_content.split('\\n')[0]
                topic_match = first_line[:40]
                top_score = scores[0]
                hit = "OK" if top_score >= RERANK_THRESHOLD else "LO"
            else:
                topic_match = "(无结果)"
                top_score = -99
                hit = "XX"

            print(f"  [{strategy_name:12s}]{extra_info} {hit} "
                  f"score={top_score:+.2f}  [{topic_match}]  ({elapsed:.1f}s)")

    print(f"\n{'=' * 60}")
    print("汇总: Top-1 命中数 / 总问题数")
    print(f"{'=' * 60}")

    for strategy_name in strategies:
        hits = 0
        total = len(TEST_QUERIES)
        for query in TEST_QUERIES:
            if strategy_name == "rewrite_loop":
                results, scores, loops = strategies[strategy_name](query)
            else:
                results, scores = strategies[strategy_name](query)
            if results and scores and scores[0] >= RERANK_THRESHOLD:
                hits += 1
        pct = (hits / total) * 100
        print(f"  {strategy_name:12s}: {hits}/{total} ({pct:.0f}%)")


# ============== 7. 单次查询使用 ==============

def search(query: str, strategy: str = "baseline"):
    """便捷接口: 用指定策略搜索"""
    docs = load_documents()
    ensemble, vector_store = init_hybrid_retriever(docs)
    rerank_tokenizer, rerank_model, device = init_reranker()
    llm = init_llm()

    strategy_map = {
        "baseline":     lambda q: full_pipeline(q, ensemble, rerank_tokenizer, rerank_model, device),
        "multi_query":  lambda q: multi_query_search(q, ensemble, rerank_tokenizer, rerank_model, device, llm),
        "hyde":         lambda q: hyde_search(q, ensemble, vector_store, rerank_tokenizer, rerank_model, device, llm),
        "rewrite_loop": lambda q: rewrite_loop_search(q, ensemble, rerank_tokenizer, rerank_model, device, llm),
    }

    if strategy not in strategy_map:
        print(f"未知策略: {strategy}, 可选: {list(strategy_map.keys())}")
        return

    fn = strategy_map[strategy]
    if strategy == "rewrite_loop":
        results, scores, loops = fn(query)
        print(f"重查次数: {loops}")
    else:
        results, scores = fn(query)

    print(f"\\n[策略: {strategy}] 查询: {query}")
    print("-" * 40)
    for i, (doc, score) in enumerate(zip(results, scores), 1):
        first_line = doc.page_content.split('\\n')[0]
        print(f"[{i}] score={score:+.2f} | {first_line[:60]}")
        print(f"    {doc.page_content[60:160]}...")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
        strategy = sys.argv[2] if len(sys.argv) > 2 else "baseline"
        search(query, strategy)
    else:
        run_ablation()
