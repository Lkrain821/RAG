"""
=========== RAG 问答 ===========
对应架构图：用户提问 → 向量化 → 检索相关文档块 → 拼接Prompt → LLM生成回答

这是 RAG 的最终输出——你问一个问题，系统从知识库找到相关内容，喂给大模型生成答案。
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
import os


# ===== 1. 检索 =====
def retrieve(query: str, vector_store: Chroma, k: int = 3):
    """
    根据用户问题，从向量数据库中检索最相关的 k 个文档块

    这个过程就是之前讲的"语义搜索"：
    用户问题 "RAG 有哪些优点？"
        ↓ Embedding
    问题向量 [0.13, -0.32, ...]
        ↓ 在向量数据库中找最近的 k 个
    返回 k 个最相关的文档块
    """
    docs = vector_store.similarity_search(query, k=k)
    return docs


# ===== 2. 拼接上下文 =====
def build_context(docs):
    """把检索到的文档块拼成一段上下文"""
    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(f"[文档片段 {i+1}]\n{doc.page_content}")
    return "\n\n".join(context_parts)


# ===== 3. 构建 Prompt =====
def build_prompt(query: str, context: str):
    """
    这是之前学过的 Prompt Engineering 实战应用！
    用"角色 + 约束 + 上下文"的结构，让模型准确回答。

    关键约束：
    - "只基于提供的文档" → 防止幻觉
    - "不知道就说不知道" → 防止编造
    - 引用来源 → 让答案可追溯
    """
    prompt = f"""你是一个知识库问答助手。请只基于以下提供的文档内容回答用户的问题。

文档内容：
{context}

用户问题：{query}

回答规则：
1. 只使用上面文档中提供的信息来回答
2. 如果文档中没有相关信息，请明确说"根据已有文档，无法回答此问题"
3. 回答时引用具体的文档片段编号
4. 回答要简洁清晰，不要添加文档中没有的内容"""
    return prompt


# ===== 4. 使用 Ollama 本地模型（免费） =====
def create_llm():
    """
    使用 Ollama 运行本地大模型，完全免费。
    需要先安装 Ollama：https://ollama.com/
    然后下载模型：ollama pull qwen2.5:0.5b  # 小模型，适合学习
    """
    try:
        llm = OllamaLLM(
            model="qwen3:4b",   # 使用你已安装的模型
            temperature=0,       # 保守模式，减少幻觉
        )
        print("[模型就绪] Ollama qwen3:4b")
        return llm
    except Exception as e:
        print(f"[提示] Ollama 未安装或未启动：{e}")
        print("下面会使用模拟模式演示完整流程……")
        return None


# ===== 5. 完整问答流程 =====
def ask(query: str, vector_store: Chroma, llm):
    """
    RAG 问答的完整流程，一步到位：
    query → retrieve → build_context → build_prompt → LLM.generate → answer
    """
    print(f"[问题] {query}")
    print("-" * 40)

    # Step A: 检索
    print("[检索] 正在检索相关文档...")
    docs = retrieve(query, vector_store, k=3)

    if not docs:
        return "未找到相关文档"

    print(f"   找到 {len(docs)} 个相关片段")
    for i, doc in enumerate(docs):
        print(f"   [{i+1}] {doc.page_content[:80]}...")

    # Step B: 构建上下文和 Prompt
    context = build_context(docs)
    prompt = build_prompt(query, context)

    # Step C: 调用 LLM 生成答案
    if llm:
        print("[生成] 正在生成回答...")
        try:
            answer = llm.invoke(prompt)
        except Exception as e:
            if "Connection" in str(e) or "11434" in str(e):
                return (
                    "无法连接到 Ollama 服务（localhost:11434）。\n\n"
                    "请先启动 Ollama：\n"
                    "  1. 打开新终端，运行 ollama serve\n"
                    "  2. 确保 qwen3:4b 模型已下载：ollama list\n"
                    "  3. 再重新运行 python chat.py"
                )
            else:
                return f"LLM 调用失败：{e}"
    else:
        # 模拟模式：展示完整的 Prompt，让你看到整个流程
        answer = (
            "(模拟模式 — 安装 Ollama 后可获得真实回答)\n\n"
            "===== 以下是传给 LLM 的完整 Prompt =====\n"
            f"{prompt}\n"
            "========================================="
        )

    return answer


# ===== 6. 跑一遍试试 =====
if __name__ == "__main__":
    print("=" * 50)
    print("RAG 问答")
    print("=" * 50)

    # 加载之前建好的向量数据库
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model,
    )

    # 初始化 LLM
    llm = create_llm()

    # 交互式问答循环
    print("输入问题开始问答，输入 quit 或 exit 退出")
    while True:
        try:
            query = input("你的问题：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break
        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not query:
            continue
        answer = ask(query, vector_store, llm)
        print(f"\n[回答]\n{answer}")
