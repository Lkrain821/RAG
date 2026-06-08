# ============== ablation_hybrid_only.py ==============
# D2-ablation:对照实验 - 验证 Rerank 单独的贡献
# 配置:chunk=200 + 混合检索 + NO Rerank
# 目的:跟 D1 (chunk=500) 和 D2 (chunk=200 + Rerank) 做三向对比
# ===============================================

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import jieba

# ============== 加载文档 ==============
loader = TextLoader("data/test.txt", encoding="utf-8")
raw_docs = loader.load()
# ⚠️ 关键:跟 D2 一样 chunk_size=200(不引入 Rerank 作为变量)
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
docs = splitter.split_documents(raw_docs)
print(f"切了 {len(docs)} 个块")

# ============== 中文分词 ==============
def chinese_tokenizer(text: str) -> list[str]:
    return list(jieba.cut(text))

# ============== 混合检索(无 Rerank)==============
bm25_retriever = BM25Retriever.from_documents(
    docs,
    preprocess_func=chinese_tokenizer
)
bm25_retriever.k = 5  # ⚠️ 跟 D1 一样只取 5(不靠 Rerank 选 Top-5)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_store = Chroma.from_documents(docs, embeddings)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)


# ============== 直接返回混合检索 Top-5(无 Rerank)==============
def hybrid_search_only(query: str, top_n: int = 5) -> list:
    return ensemble_retriever.invoke(query)[:top_n]


# ============== 测试 ==============
if __name__ == "__main__":
    print("=" * 50)
    print("Ablation: chunk=200 + 混合检索(无 Rerank)")
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
        results = hybrid_search_only(query, top_n=5)
        for i, doc in enumerate(results, 1):
            first_line = doc.page_content.split('\n')[0]
            print(f"[{i}] [{first_line[:30]}] {doc.page_content[30:130]}...")
