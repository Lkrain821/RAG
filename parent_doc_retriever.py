"""
=========== D8: Parent-Document Retriever ===========
Stage 3 (D8): "细检索粗返回" 的 Parent-Document Retriever

核心思想:
  - 检索用细粒度子块（~200字，查得准）
  - 返回用粗粒度父块（~1500字，上下文够）
  - 典型应用: 法律/医疗/金融文档，需要看到完整条款/病历/报表

对比:
  - 普通检索: 返回碎片化的 200 字小块，上下文不完整
  - ParentDoc: 返回 1500 字完整段落，上下文充足

使用方式:
  python parent_doc_retriever.py                    # A/B 对比测试
  python parent_doc_retriever.py "什么是RAG"        # 单查询
"""

import os
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore


# ============== 配置 ==============

DATA_PATH = "data/test.txt"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
PARENT_CHUNK_SIZE = 1500   # 父块: ~750 中文字，上下文充足
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 200     # 子块: ~100 中文字，检索精准
CHILD_CHUNK_OVERLAP = 30
TOP_K = 5

# 中文分隔符
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", " ", ""]


# ============== 1. 加载文档 ==============

def load_documents():
    """加载 data/test.txt"""
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    docs = loader.load()
    print(f"[加载] {DATA_PATH}: {len(docs)} 个文档")
    return docs


# ============== 2. 构建检索器 ==============

def create_parent_doc_retriever(documents):
    """
    构建 Parent-Document Retriever

    工作原理:
    1. 父块切分器: 把原始文档切成 ~1500字 的大块（parent chunks）
    2. 子块切分器: 把每个父块再切成 ~200字 的小块（child chunks）
    3. 子块的 embedding 存入 ChromaDB（检索用）
    4. 父块的原文存入 InMemoryStore（返回用）
    5. 子块 metadata 中记录 parent_id，关联到对应的父块

    检索时:
    query → embedding → 在子块中找到最相关的 top-k
    → 通过 parent_id 回溯 → 返回对应的父块原文
    """
    # 切分器
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=SEPARATORS,
        add_start_index=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=SEPARATORS,
        add_start_index=True,
    )

    # 向量库（存子块的 embedding）
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name="parent_doc_children",
        embedding_function=embedding_model,
    )

    # 文档存储（存父块的原文）
    store = InMemoryStore()

    # 组装 ParentDocumentRetriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 添加文档（内部自动做: 父切分 → 子切分 → 存向量 → 存docstore）
    retriever.add_documents(documents)

    return retriever, store


def create_baseline_retriever(documents):
    """普通检索器: 直接用 200字 子块建索引，检索也返回子块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma.from_documents(chunks, embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": TOP_K}), chunks


def create_parent_doc_pipeline():
    """
    创建供 eval.py 调用的 pipeline 函数。
    返回签名: (query: str) -> (list[Document], list[float])
    与 eval.py 中其他 pipeline 兼容。
    """
    documents = load_documents()
    retriever, _store = create_parent_doc_retriever(documents)

    def pipeline(query: str):
        docs = retriever.invoke(query)
        # ParentDocRetriever 不返回分数，用 0.0 填充
        scores = [0.0] * len(docs)
        return docs, scores

    return pipeline


# ============== 3. A/B 对比测试 ==============

TEST_QUERIES = [
    "什么是 RAG？它能解决什么问题？",
    "Embedding 是怎么把文本变成向量的？",
    "RRF 公式是什么？",
    "BGE-Reranker 怎么用？",
    "Agentic RAG 的核心思想是什么？",
]


def compare_retrievers(test_queries: list[str]):
    """对比普通 retriever vs ParentDoc retriever"""
    documents = load_documents()

    print("\n[初始化] 普通检索器 ...")
    baseline_retriever, baseline_chunks = create_baseline_retriever(documents)
    print(f"  -> {len(baseline_chunks)} 个子块")

    print("[初始化] ParentDoc 检索器 ...")
    parent_retriever, store = create_parent_doc_retriever(documents)

    # 查看 docstore 中有多少父块
    parent_count = len(store.mget(list(store.yield_keys())))
    print(f"  -> {parent_count} 个父块")

    print(f"\n{'=' * 60}")
    print(f"  A/B 对比: 普通检索 vs Parent-Document Retriever")
    print(f"{'=' * 60}")

    baseline_total_len = 0
    parent_total_len = 0
    count = 0

    for q_idx, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"Q{q_idx}: {query}")
        print(f"{'─' * 60}")

        # 普通检索
        t0 = time.time()
        baseline_docs = baseline_retriever.invoke(query)
        baseline_time = time.time() - t0

        # ParentDoc 检索
        t0 = time.time()
        parent_docs = parent_retriever.invoke(query)
        parent_time = time.time() - t0

        # 展示 top-1 结果对比
        print(f"\n  [普通检索] top-1 ({len(baseline_docs[0].page_content) if baseline_docs else 0}字, "
              f"{baseline_time:.3f}s):")
        if baseline_docs:
            preview = baseline_docs[0].page_content[:120].replace("\n", " ")
            print(f"    {preview}...")
            baseline_total_len += sum(len(d.page_content) for d in baseline_docs)

        print(f"\n  [ParentDoc] top-1 ({len(parent_docs[0].page_content) if parent_docs else 0}字, "
              f"{parent_time:.3f}s):")
        if parent_docs:
            preview = parent_docs[0].page_content[:120].replace("\n", " ")
            print(f"    {preview}...")
            parent_total_len += sum(len(d.page_content) for d in parent_docs)

        count += 1

    # 汇总
    print(f"\n{'=' * 60}")
    print("汇总对比")
    print(f"{'=' * 60}")

    if count > 0:
        baseline_avg = baseline_total_len / (count * TOP_K) if baseline_total_len else 0
        parent_avg = parent_total_len / (count * TOP_K) if parent_total_len else 0

        print(f"{'指标':16s} | {'普通检索':>10s} | {'ParentDoc':>10s}")
        print("-" * 42)
        print(f"{'平均返回长度(字)':16s} | {baseline_avg:>10.0f} | {parent_avg:>10.0f}")
        print(f"{'上下文完整性':16s} | {'碎片化':>10s} | {'完整段落':>10s}")

        if parent_avg > baseline_avg * 1.5:
            print(f"\n结论: ParentDoc 返回的文档平均长度是普通检索的 {parent_avg/baseline_avg:.1f} 倍")
            print("      上下文更完整，适合需要'看到完整段落'的场景（法律/医疗/金融）")
        else:
            print(f"\n注意: 当前数据量较小，ParentDoc 优势不明显")
            print("      长文档（>5页 PDF）下差异更显著")


# ============== 4. 单查询 ==============

def search(query: str, mode: str = "parent"):
    """便捷搜索接口"""
    documents = load_documents()

    if mode == "parent":
        retriever, _ = create_parent_doc_retriever(documents)
        label = "ParentDoc"
    else:
        retriever, _ = create_baseline_retriever(documents)
        label = "普通检索"

    print(f"\n[{label}] 查询: {query}")
    print("-" * 40)

    docs = retriever.invoke(query)
    for i, doc in enumerate(docs, 1):
        content_len = len(doc.page_content)
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"[{i}] ({content_len}字) {preview}...")
        print()


# ============== 5. 主入口 ==============

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print("=" * 60)
        print("Parent-Document Retriever 单查询")
        print("=" * 60)
        search(query, mode="parent")
    else:
        print("=" * 60)
        print("D8: Parent-Document Retriever 对比测试")
        print("=" * 60)
        compare_retrievers(TEST_QUERIES)
