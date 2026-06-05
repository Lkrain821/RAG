"""
=========== 文档上传工具 ===========
支持 TXT / PDF 文件上传到知识库

用法：
  python upload.py <文件路径>
  python upload.py                  # 交互式输入路径
  python upload.py --rebuild        # 重建整个向量库（清空旧数据）
"""

import sys
import os
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from doc_loader import load_txt, load_pdf, split_documents


# ===== 1. 识别文件类型 =====
def load_file(filepath: str):
    """根据扩展名自动选择加载方式"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        return load_txt(filepath)
    elif ext == ".pdf":
        return load_pdf(filepath)
    else:
        print(f"[错误] 不支持的文件类型: {ext}，仅支持 .txt 和 .pdf")
        return None


# ===== 2. 完整上传流程 =====
def upload(filepath: str, chunk_size=500, chunk_overlap=50, rebuild=False):
    """
    加载 → 分割 → 向量化 → 存入 ChromaDB
    """
    # Step 1: 加载
    print(f"[上传] 正在加载: {filepath}")
    docs = load_file(filepath)
    if not docs:
        return

    # Step 2: 分割
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 3: 创建 Embedding 模型
    print("[Embedding] 正在加载模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Step 4: 存入 ChromaDB
    if rebuild:
        # 清空旧数据重建
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            print("[重建] 已清空旧的向量库")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="./chroma_db",
        )
    else:
        # 追加到现有库
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="./chroma_db",
        )

    print(f"[完成] 已将 {len(chunks)} 个文本块存入向量库")
    print(f"[提示] 现在可以运行 python chat.py 开始问答")
    return vector_store


# ===== 3. 命令行入口 =====
if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv

    # 获取文件路径（命令行参数）
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        filepath = args[0]
    else:
        filepath = ""

    # 循环直到用户提供有效文件
    while True:
        if not filepath:
            filepath = input("请输入要上传的文件路径: ").strip()
            filepath = filepath.strip('"').strip("'")

        if not os.path.exists(filepath):
            print(f"[错误] 文件不存在: {filepath}")
            filepath = ""
            continue

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in (".txt", ".pdf"):
            print(f"[错误] 不支持的文件类型: {ext}，仅支持 .txt 和 .pdf")
            filepath = ""
            continue

        break

    upload(filepath, rebuild=rebuild)
