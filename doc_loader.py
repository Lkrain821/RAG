"""
=========== RAG 文档加载与文本分割 ===========
对应架构图：文档(PDF/TXT) → 文本分割(Chunking)

核心概念：
- Chunk（块）：把长文档切成小段，每段是一个"知识点"
- Chunk size：每段最多多少字。太大 → 检索不精准；太小 → 缺少上下文
- Chunk overlap：相邻两段重叠多少字。防止一个完整句子被切在两段中间
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===== 1. 加载文档 =====
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
import csv

# 加载一个 txt 文件
def load_txt(filepath: str):
    """加载纯文本文件，返回一个 Document 对象列表"""
    loader = TextLoader(filepath, encoding="utf-8")
    documents = loader.load()
    print(f"[加载完成] 共 {len(documents)} 个文档")
    return documents

# 加载一个 PDF 文件
def load_pdf(filepath: str):
    """加载 PDF 文件"""
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    print(f"[加载完成] 共 {len(documents)} 页")
    return documents

# 加载一个 CSV 文件
def load_csv(filepath: str):
    """加载 CSV 表格，转为结构化自然语言文本，便于 Embedding 和检索"""
    with open(filepath, encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    # 生成摘要（放最前面，便于检索"这个表有哪些列"）
    header_str = "」「".join(headers)
    summary = (
        f"表格摘要：此表格包含 {len(headers)} 列，"
        f"分别为「{header_str}」。共 {len(rows)} 行数据。"
    )

    # 逐行转结构化文本（"表头: 值"格式，Embedding 能理解列间关系）
    lines = [summary, "=" * 50]
    for i, row in enumerate(rows):
        parts = [f"{headers[j]}: {row[j]}" for j in range(len(headers))]
        lines.append(f"第{i+1}行 — " + " | ".join(parts))

    text = "\n".join(lines)
    doc = Document(page_content=text, metadata={"source": filepath})
    print(f"[加载完成] CSV 表格：{len(headers)} 列，{len(rows)} 行，已转为自然语言文本")
    return [doc]


# ===== 2. 文本分割 =====
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    把长文档切成小块

    参数说明：
    - chunk_size=500：每块最多 500 个字符
      （中文约 250 字，因为中文一个字符 ≈ 2 个英文字符）
    - chunk_overlap=50：相邻两块重叠 50 个字符
      举例：块1 是 [0:500]，块2 是 [450:950]，块3 是 [900:1400]……
      重叠部分保证知识不会因切分而断裂
    """
    # Recursive 的意思：按优先级 "\n\n" → "\n" → " " → "" 来切分
    # 先尝试按段落切，再按句子切，最后按字符切
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    print(f"[分割完成] {len(documents)} 个文档 → {len(chunks)} 个文本块")
    print(f"[示例] 第一块前 100 字：{chunks[0].page_content[:100]}...")
    return chunks


# ===== 3. 跑一遍试试 =====
if __name__ == "__main__":
    # 先创建一个测试文档
    test_file = "/d/Pythoncode/RAG/data/test.txt"
    import os
    if not os.path.exists(test_file):
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(
                "RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的技术。\n\n"
                "它的核心思想是：在生成回答之前，先从外部知识库中检索相关文档，然后将检索到的内容作为上下文输入给大语言模型。\n\n"
                "这样做的好处有三个：第一，解决大模型的知识过时问题；第二，让大模型能够访问私有数据；第三，有效减少幻觉。\n\n"
                "RAG 的工作流程分为两个阶段：离线阶段进行文档加载、文本分割和向量化存储；在线阶段进行查询向量化、相似度检索和答案生成。\n\n"
                "向量化（Embedding）是 RAG 的核心环节之一，它将文本转换为高维向量，使得语义相似的文本在向量空间中距离更近。\n\n"
                "常用的 Embedding 模型包括 OpenAI 的 text-embedding-3-small、智源的 BGE 系列、以及开源的 sentence-transformers。\n\n"
                "选择合适的 Chunk 大小非常关键——太大会引入无关信息，干扰模型判断；太小则会丢失上下文，导致检索遗漏。\n\n"
                "一般来说，中文文本的 Chunk 大小推荐在 300-800 字符之间，重叠部分建议占 Chunk 大小的 10%-20%。"
            )

    print("=" * 50)
    print("文档加载与文本分割")
    print("=" * 50)

    docs = load_txt(test_file)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)

    print("\n[所有分割块一览]")
    for i, chunk in enumerate(chunks):
        print(f"--- 块 {i+1}（长度 {len(chunk.page_content)} 字）---")
        print(chunk.page_content[:150] + "...")
        print()
