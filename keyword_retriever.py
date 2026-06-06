"""
=========== 关键词检索 + 混合检索 ===========

演示三种检索方式：
1. 语义检索（你已有的）：问题向量化 → 余弦相似度
2. 关键词检索（新增）：BM25 算法，匹配字面
3. 混合检索（新增）：语义 + 关键词，取交集/加权
"""

from rank_bm25 import BM25Okapi
import jieba


# ===== 1. 中文分词 =====
def tokenize(text: str):
    """把中文句子切成词"""
    return list(jieba.cut(text))


# ===== 2. 关键词检索（BM25）=====
class KeywordRetriever:
    """
    BM25 关键词检索器

    原理：
    - 一篇文档中，某个词出现次数多 → 这个词在这篇文档里重要（词频 TF）
    - 但"的"/"了"这种词在每篇都有 → 没法靠它区分文档，降低权重（逆文档频率 IDF）
    """

    def __init__(self, documents: list[str]):
        # 把所有文档分词
        self.documents = documents
        self.tokenized_docs = [tokenize(doc) for doc in documents]
        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, k: int = 3):
        """返回最相关的 k 篇文档，带分数"""
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # 按分数排序
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]
        return [(self.documents[i], score) for i, score in ranked]


# ===== 3. 混合检索 =====
class HybridRetriever:
    """
    混合语义检索 + 关键词检索的结果

    策略：两个检索器各搜 k 个结果，汇总去重后重新排序
    """

    def __init__(self, vector_store, keyword_retriever: KeywordRetriever):
        self.vector_store = vector_store
        self.keyword = keyword_retriever

    def search(self, query: str, k: int = 3):
        # 语义检索 k 个
        semantic_docs = self.vector_store.similarity_search(query, k=k)
        # 关键词检索 k 个
        keyword_results = self.keyword.search(query, k=k)

        # 简单混合：去重汇总，取前 k 个
        seen = set()
        merged = []

        # 语义结果优先
        for doc in semantic_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)

        # 关键词结果补充（不重复的才加）
        for content, _ in keyword_results:
            if content not in seen and len(merged) < k * 2:
                seen.add(content)
                merged.append(content)

        return merged[:k]


# ===== 4. 对比演示 =====
if __name__ == "__main__":
    # 准备文档
    documents = [
        "RAG 是一种结合信息检索与文本生成的技术，可以有效减少大模型的幻觉问题",
        "RAG 的核心流程是：用户提问后，先检索相关文档，再拼接 Prompt 交给 LLM 生成答案",
        "Embedding 模型可以将文本转换为向量，语义相近的文本向量距离也近",
        "苹果手机最新款搭载了 A18 芯片，续航时间达到 30 小时",
        "红富士苹果是一种常见的水果，富含维生素 C 和膳食纤维",
        "BM25 是一种经典的关键词检索算法，基于词频和逆文档频率",
    ]

    print("=" * 60)
    print("文档库：")
    for i, doc in enumerate(documents):
        print(f"  [{i}] {doc}")

    # 关键词检索
    kr = KeywordRetriever(documents)

    print("\n" + "=" * 60)
    print("【关键词检索】问题：苹果")
    print("-" * 60)
    print("流程: 分词 -> 匹配包含[苹果]的文档 -> BM25 评分排序")
    print("-" * 60)
    for content, score in kr.search("苹果", k=5):
        print(f"  分数 {score:.4f}  →  {content}")

    print("\n" + "=" * 60)
    print("【关键词检索】问题：RAG 是什么")
    print("-" * 60)
    for content, score in kr.search("RAG 是什么", k=5):
        print(f"  分数 {score:.4f}  →  {content}")

    print("\n" + "=" * 60)
    print("对比总结")
    print("-" * 60)
    print("关键词检索: 问[苹果] -> 手机和水果都搜到, 靠 BM25 评分区分")
    print("语义检索:   问[苹果] -> 靠上下文语义, 自动区分手机 vs 水果")
    print("混合检索:   两种结果合并去重 -> 覆盖面更广, 不遗漏")
