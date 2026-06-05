"""
=========== RAG 向量化与存储 ===========
对应架构图：文本分割 → 向量化(Embedding) → 向量数据库(Vector Store)

核心概念：
- Embedding（向量化）：把一段文字变成一串数字（向量）
  语义相近的文本 → 向量之间的距离也近
- Vector Store（向量数据库）：存储这些向量，支持"相似度搜索"
  你问一个问题 → 它找到最相关的文档块
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# ===== 1. 创建 Embedding 模型 =====
# 使用智源的 BGE 中文模型，免费、本地运行、中文效果好
# 第一次运行会自动下载模型文件（约 400MB），请耐心等待
def create_embedding_model():
    """
    BGE (BAAI General Embedding) 是智源研究院开源的中文向量模型
    bge-small-zh-v1.5：轻量版，速度快，适合学习和原型开发
    """
    model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},  # 用 CPU 跑，不用 GPU
        encode_kwargs={"normalize_embeddings": True},  # 归一化，便于计算相似度
    )
    print("[模型就绪] BGE-small-zh-v1.5 Embedding 模型加载完成")
    return model


# ===== 2. 创建向量数据库 =====
def create_vector_store(chunks, embedding_model, persist_dir="./chroma_db"):
    """
    把文本块向量化后存入 ChromaDB

    参数：
    - chunks：doc_loader 分割好的文本块
    - embedding_model：向量化模型
    - persist_dir：向量数据库存储路径

    原理图解：
    文本块1: "RAG是一种技术..." → Embedding → [0.12, -0.34, 0.56, ...] (512维)
    文本块2: "它的核心思想是..." → Embedding → [0.13, -0.32, 0.55, ...] (512维)
    文本块3: "向量化是RAG的..."   → Embedding → [0.11, -0.35, 0.58, ...] (512维)
    ...

    当你提问 "什么是RAG？" 时：
    提问向量:  [0.12, -0.33, 0.57, ...]
                        ↓ 计算余弦相似度
    文本块1:  相似度 0.95 ← 最相关！
    文本块3:  相似度 0.82
    文本块2:  相似度 0.45
    """
    # from_documents 会做两件事：
    # 1. 把每个 chunk 向量化
    # 2. 存入 ChromaDB 并建立索引
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    print(f"[存储完成] {len(chunks)} 个文本块已存入 {persist_dir}")
    return vector_store


# ===== 3. 加载已有的向量数据库 =====
def load_vector_store(embedding_model, persist_dir="./chroma_db"):
    """从磁盘加载之前存好的向量数据库（不用每次重建）"""
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )
        print(f"[加载完成] 已从 {persist_dir} 加载向量数据库")
        return vector_store
    else:
        print(f"[提示] {persist_dir} 不存在或为空，请先运行 create_vector_store")
        return None


# ===== 4. 跑一遍试试 =====
if __name__ == "__main__":
    from doc_loader import load_txt, split_documents

    print("=" * 50)
    print("向量化与存储")
    print("=" * 50)

    # 加载和分割
    test_file = "/d/Pythoncode/RAG/data/test.txt"
    if not os.path.exists(test_file):
        print("[错误] 请先运行 doc_loader.py 生成测试数据")
        exit(1)

    docs = load_txt(test_file)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)

    # 创建 Embedding 模型
    embedding_model = create_embedding_model()

    # 存入向量数据库
    vector_store = create_vector_store(chunks, embedding_model)

    # 试试搜索
    question = "什么是 RAG？"
    print(f"\n[搜索测试] 问题：{question}")
    results = vector_store.similarity_search(question, k=3)

    for i, doc in enumerate(results):
        print(f"\n相关度 #{i+1}：")
        print(doc.page_content[:200])
