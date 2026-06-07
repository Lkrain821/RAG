# ============== hybrid_retriever.py ==============
# D1 任务:混合检索骨架
# 你需要做的是下面标了 [填空] 的 5 个动作
# ===============================================

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import jieba

# ============== [填空 1] 准备文档 ==============
# 用你已经有的 doc_loader.py 里的逻辑
# 提示:用 TextLoader 加载 data/test.txt,然后 split_documents
# 你 doc_loader.py 里的代码逻辑,抄过来就行
loader = TextLoader("data/test.txt", encoding="utf-8")
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)
print(f"切了 {len(docs)} 个块")
# ==============================================
# ==============================================


# ============== [填空 2] 中文分词器 ==============
# RRF 公式要求每个检索器返回排名,BM25 需要分词
# 提示:用 jieba.cut 切,转成 list
def chinese_tokenizer(text: str) -> list[str]:
    return list(jieba.cut(text))
# ==============================================


# ============== [填空 3] 初始化 BM25 ==============
# 用 BM25Retriever.from_documents 把 chunks 喂进去
# 提示:传 chunks 和 preprocess_func 参数
bm25_retriever = BM25Retriever.from_documents(
    docs,
    preprocess_func=chinese_tokenizer
)
bm25_retriever.k = 5
# ==============================================


# ============== [填空 4] 初始化向量检索 ==============
# 用 Chroma.from_documents + BGE embedding
# 提示:用你已经熟悉的 BAAI/bge-small-zh-v1.5
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_store = Chroma.from_documents(docs, embeddings)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
# ==============================================


# ============== [填空 5] RRF 融合 ==============
# 把两个 retriever 装进 EnsembleRetriever
# 提示:weights=[0.5, 0.5] 表示等权
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
# ==============================================


# ============== 测试(不用改)==============
if __name__ == "__main__":
    print("=" * 50)
    print("混合检索测试(BM25 + 向量 + RRF)")
    print("=" * 50)
    
    # 测试查询:5 条针对不同主题的问题(用于 A/B 对比)
    test_queries = [
        "什么是 RAG?它能解决什么问题?",
        "RAG 的工作流程分哪两个阶段?",
        "Embedding 是怎么把文本变成向量的?",
        "Chroma 和 Milvus 有什么区别?",
        "RRF 公式是什么?为什么用排名不用分数?",
        "BGE-Reranker 怎么用?",
        "RAGAS 的 4 个核心指标是什么?",
        "Agentic RAG 的核心思想是什么?",
    ]
    
    for query in test_queries:
        print(f"\n[问题] {query}")
        print("-" * 40)
        docs = ensemble_retriever.invoke(query)
        for i, doc in enumerate(docs, 1):
            # 显示段落来源(从 page_content 第一行的 # 主题里抓)
            first_line = doc.page_content.split('\n')[0]
            print(f"[{i}] [{first_line[:30]}] {doc.page_content[30:130]}...")