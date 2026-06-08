# ============== rerank_hybrid.py ==============
# D2 任务:在 D1 混合检索基础上加 BGE-Reranker 精排
# 实现方式:用 HuggingFace transformers 直接加载 BGE-Reranker
# (绕开 FlagEmbedding 1.4.0 与 transformers 5.x 的兼容问题)
# ===============================================

import os
import torch

# 抑制 huggingface 缓存 symlink 警告(Windows 上不痛不痒)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import jieba

# ============== 加载文档(D1 已有,直接复用)==============
loader = TextLoader("data/test.txt", encoding="utf-8")
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)  # chunk 切细
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


# ============== 初始化 BGE-Reranker(Cross-Encoder)==============
# 用 transformers 直接加载,绕开 FlagEmbedding 的 prepare_for_model 兼容问题
# BGE-Reranker 是 Cross-Encoder,把 query+doc 一起编码,比 Bi-Encoder 向量检索更精准
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

print(f"正在加载 Reranker 模型:{RERANK_MODEL_NAME} ...")
rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_NAME)
rerank_model.eval()  # 推理模式

# 自动选设备:有 GPU 用 GPU,没 GPU 用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rerank_model.to(device)
print(f"Reranker 模型已加载到 {device}")
# ==============================================


# ============== Rerank 精排函数 ==============
# 输入:query 字符串 + 候选 docs 列表
# 输出:rerank 后的 docs 列表(Top-N)
def rerank_docs(query: str, candidates: list, top_n: int = 5) -> list:
    if not candidates:
        return []
    
    # 1. 构造 [query, doc_content] 配对
    pairs = [[query, doc.page_content] for doc in candidates]
    
    # 2. tokenizer 编码(自动 padding + truncation)
    with torch.no_grad():
        inputs = rerank_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # 3. 模型推理,拿到 relevance 分数
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
    
    # 4. 按分数降序排,取 Top-N
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in sorted_indices[:top_n]]
# ==============================================


# ============== 拼装完整流水线 ==============
# 流水线:query → 混合检索(召回 20)→ Rerank 精排(取 Top-5)
def hybrid_search(query: str, top_n: int = 5) -> list:
    candidates = ensemble_retriever.invoke(query)
    return rerank_docs(query, candidates, top_n)
# ==============================================


# ============== 测试 ==============
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
