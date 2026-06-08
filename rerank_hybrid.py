# ============== rerank_hybrid.py ==============
# D2 任务:在 D1 混合检索基础上加 BGE-Reranker 精排
# 你需要做的是下面标了 [填空] 的 3 个动作
# ===============================================

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import FlagReranker
import jieba

# ============== 加载文档(D1 已有,直接复用)==============
loader = TextLoader("data/test.txt", encoding="utf-8")
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)
print(f"切了 {len(docs)} 个块")

# ============== 中文分词(同 D1)==============
def chinese_tokenizer(text: str) -> list[str]:
    return list(jieba.cut(text))

# ============== 混合检索(D1 已有)==============
bm25_retriever = BM25Retriever.from_documents(
    docs,
    preprocess_func=chinese_tokenizer
)
bm25_retriever.k = 20  # D2 关键改动:先粗召回 20 个

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_store = Chroma.from_documents(docs, embeddings)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 20})  # D2 关键改动

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
# ==============================================


# ============== [填空 1] 初始化 Reranker ==============
# BGE-Reranker 是 Cross-Encoder,比 Bi-Encoder 向量检索更精准但更慢
# 第一次跑会下载模型(约 1.1GB),后续会缓存
# 提示:用 FlagReranker 类
reranker = None  # ← 你来填
# ==============================================


# ============== [填空 2] Rerank 函数 ==============
# 输入:query 字符串 + 候选 docs 列表
# 输出:rerank 后的 docs 列表(Top-N)
# 提示:
#   1. 把每个 doc 转成 "query + doc_content" 配对
#   2. reranker.compute_score() 算分数
#   3. 按分数降序排,取前 top_n 个
def rerank_docs(query: str, candidates: list, top_n: int = 5) -> list:
    """对初检索结果做精排,返回 Top-N 文档列表"""
    return None  # ← 你来填
# ==============================================


# ============== [填空 3] 拼装完整流水线 ==============
# 流水线:query → retrieve(20) → rerank → top-5
# 提示:ensemble_retriever.invoke(query) 拿 20 个,再 rerank_docs 取 5 个
def hybrid_search(query: str, top_n: int = 5) -> list:
    """完整流水线:混合检索 + Rerank 精排"""
    return None  # ← 你来填
# ==============================================


# ============== 测试(不用改)==============
if __name__ == "__main__":
    print("=" * 50)
    print("混合检索 + BGE-Reranker 精排 测试")
    print("=" * 50)
    
    test_queries = [
        "什么是 RAG?它能解决什么问题?",
        "Embedding 是怎么把文本变成向量的?",
        "RRF 公式是什么?为什么用排名不用分数?",
        "BGE-Reranker 怎么用?",
        "RAGAS 的 4 个核心指标是什么?",
        "Agentic RAG 的核心思想是什么?",
    ]
    
    for query in test_queries:
        print(f"\n[问题] {query}")
        print("-" * 40)
        results = hybrid_search(query, top_n=5)
        for i, doc in enumerate(results, 1):
            first_line = doc.page_content.split('\n')[0]
            print(f"[{i}] [{first_line[:30]}] {doc.page_content[30:130]}...")
