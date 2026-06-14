"""
=========== RAGAS 评估 Pipeline ===========
Stage 2-3 (D4-D9): RAG 评估体系

对 RAG 系统进行 4 指标评估:
  - context_precision: 检索结果中相关内容的占比
  - context_recall:   应该召回的有没有召回来
  - faithfulness:     答案是否完全基于检索内容（防幻觉）
  - answer_relevancy: 答案与问题的相关度

支持三种模式:
  - baseline: naive RAG (纯向量检索, 无 rerank, 无 query 改写)
  - improved: Stage 1 完整流水线 (混合检索 + Rerank + rewrite_loop)
  - stage3:   Stage 3 数据层优化 (语义切块 + ParentDoc + Rerank)

使用方式:
  python eval.py baseline    # 跑 baseline 评估
  python eval.py improved    # 跑 Stage1 改造后评估
  python eval.py stage3      # 跑 Stage3 语义切块+ParentDoc 评估
  python eval.py compare     # 三方对比报告
"""

import json
import os
import sys
import time
from typing import Optional

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


# ============== 配置 ==============

# LLM-as-Judge 配置 (RAGAS 需要强 LLM 做评判)
JUDGE_API_KEY = os.environ.get("RAGAS_API_KEY", "sk-aef72c31fcd241b7adb9c6a9630f2f28")
JUDGE_BASE_URL = os.environ.get("RAGAS_BASE_URL", "https://api.deepseek.com/v1")
JUDGE_MODEL = os.environ.get("RAGAS_MODEL", "deepseek-v4-flash")

# RAG 回答生成 LLM (本地 Ollama)
ANSWER_LLM_MODEL = "qwen3:4b"

# 评估数据
EVAL_DATA_PATH = "data/eval_qa.jsonl"

# 检索/文档配置
DATA_PATH = "data/test.txt"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
HYBRID_K = 20
TOP_N = 5

# Stage 3 新增配置
PARENT_CHUNK_SIZE = 1500       # 父块大小（~750 中文字）
CHILD_CHUNK_SIZE = 200         # 子块大小（与 CHUNK_SIZE 一致）
CHILD_CHUNK_OVERLAP = 30
SEMANTIC_BREAKPOINT_TYPE = "percentile"
SEMANTIC_BREAKPOINT_AMOUNT = 95


# ============== 1. 加载评估数据集 ==============

def load_eval_dataset(path: str = EVAL_DATA_PATH) -> list[dict]:
    """加载 eval QA jsonl"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"[数据集] 加载 {len(samples)} 条评估问答对")
    return samples


# ============== 2. RAG 回答生成 ==============

def generate_answer_with_context(
    query: str,
    retrieved_docs: list,
    llm_answer
) -> str:
    """基于检索到的文档生成答案"""
    if not retrieved_docs:
        return "根据已有文档，无法回答此问题。"

    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(f"[文档片段 {i+1}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    prompt = f"""你是一个知识库问答助手。请只基于以下提供的文档内容回答用户的问题。

文档内容：
{context}

用户问题：{query}

回答规则：
1. 只使用上面文档中提供的信息来回答
2. 如果文档中没有相关信息，请明确说"根据已有文档，无法回答此问题"
3. 回答要简洁清晰，不要添加文档中没有的内容"""

    try:
        answer = llm_answer.invoke(prompt)
        return answer.strip()
    except Exception as e:
        print(f"  [警告] LLM 回答生成失败: {e}")
        return f"(生成失败: {e})"


# ============== 3. 构建 RAGAS EvaluationDataset ==============

def build_ragas_samples(
    eval_samples: list[dict],
    run_pipeline_fn,
    llm_answer
) -> list:
    """
    对每条 eval 问题跑 RAG pipeline, 构建 SingleTurnSample 列表。
    """
    from ragas import SingleTurnSample

    ragas_samples = []
    total = len(eval_samples)

    for idx, sample in enumerate(eval_samples):
        query = sample["question"]
        ground_truth = sample["ground_truth"]

        retrieved_docs, _scores = run_pipeline_fn(query)
        contexts = [doc.page_content for doc in retrieved_docs]
        answer = generate_answer_with_context(query, retrieved_docs, llm_answer)

        ragas_samples.append(SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts,
            response=answer,
            reference=ground_truth,
        ))

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"  [{idx+1}/{total}] 已处理...")

    return ragas_samples


# ============== 4. RAGAS 评估执行 ==============

def run_ragas_eval(ragas_samples: list, label: str) -> dict:
    """跑 RAGAS 4 指标评估"""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        context_precision, context_recall, faithfulness, answer_relevancy,
    )
    from openai import OpenAI
    from ragas.llms import llm_factory
    from langchain_huggingface import HuggingFaceEmbeddings as LcEmbeddings

    print(f"\n[评估] {label}: {len(ragas_samples)} 条样本")
    print(f"  Judge: {JUDGE_MODEL} @ {JUDGE_BASE_URL}")

    client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
    judge_llm = llm_factory(JUDGE_MODEL, client=client)

    # Embeddings for answer_relevancy (use LangChain style, evaluate() accepts it)
    ragas_embeddings = LcEmbeddings(model_name=EMBEDDING_MODEL)

    dataset = EvaluationDataset(samples=ragas_samples)

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    print(f"  开始评估 (4 指标 * {len(ragas_samples)} 条)...")
    t0 = time.time()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=ragas_embeddings,
        show_progress=True,
    )

    elapsed = time.time() - t0
    print(f"  完成, 耗时 {elapsed:.1f}s")

    scores = {}
    for m in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        try:
            val = result[m]
            # result[m] returns a list of per-sample scores; take mean
            if isinstance(val, list) and len(val) > 0:
                scores[m] = round(float(sum(val) / len(val)), 4)
            else:
                scores[m] = round(float(val), 4)
        except (KeyError, TypeError, ValueError):
            scores[m] = None

    return {"label": label, "scores": scores, "elapsed_s": round(elapsed, 1)}


# ============== 5. Pipeline 工厂 ==============

def create_baseline_pipeline():
    """Naive RAG: 纯向量检索, 无 rerank, 无 query 改写"""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("\n[初始化] Baseline Pipeline (纯向量检索)...")
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_documents(raw_docs)
    print(f"  文档: {len(docs)} 个块")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_N})

    def pipeline(query: str):
        docs_result = vector_retriever.invoke(query)
        return docs_result, [0.0] * len(docs_result)

    return pipeline


def create_improved_pipeline():
    """Stage 1 完整流水线: 混合检索 + Rerank + rewrite_loop"""
    import torch
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import OllamaLLM
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import jieba

    print("\n[初始化] Improved Pipeline (Stage 1)...")

    loader = TextLoader(DATA_PATH, encoding="utf-8")
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_documents(raw_docs)
    print(f"  文档: {len(docs)} 个块")

    def chinese_tokenizer(text: str) -> list[str]:
        return list(jieba.cut(text))

    bm25 = BM25Retriever.from_documents(docs, preprocess_func=chinese_tokenizer)
    bm25.k = HYBRID_K
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": HYBRID_K})
    ensemble = EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.5, 0.5])

    print(f"  加载 Reranker: {RERANK_MODEL} ...")
    rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
    rerank_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rerank_model.to(device)
    print(f"    -> device={device}")

    rewrite_llm = OllamaLLM(model=ANSWER_LLM_MODEL, temperature=0)

    def rerank_docs(query, candidates):
        if not candidates:
            return [], []
        pairs = [[query, doc.page_content] for doc in candidates]
        with torch.no_grad():
            inputs = rerank_tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            scores = rerank_model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
        sorted_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_pairs[:TOP_N]], [p[1] for p in sorted_pairs[:TOP_N]]

    def rewrite_loop_search(query):
        candidates = ensemble.invoke(query)
        results, scores = rerank_docs(query, candidates)

        if results and scores and scores[0] >= 0.0:
            return results, scores

        rewrite_prompt = f"""你是一个检索专家。用户的原始问题是: {query}
检索系统未能找到相关文档。请将这个问题改写得更具体、更贴近文档中的术语。
只输出改写后的问题,不要加任何解释。"""

        for _attempt in range(2):
            try:
                rewritten = rewrite_llm.invoke(rewrite_prompt).strip()
                candidates = ensemble.invoke(rewritten)
                results, scores = rerank_docs(rewritten, candidates)
                if results and scores and scores[0] >= 0.0:
                    return results, scores
            except Exception:
                pass
        return results, scores

    return rewrite_loop_search


def create_stage3_pipeline():
    """
    Stage 3 完整流水线: 语义切块 + ParentDoc Retriever + 混合检索 + Rerank + rewrite_loop

    与 improved 的区别:
    1. 切块策略: RecursiveTextSplitter → SemanticChunker（语义感知切块）
    2. 检索方式: 普通向量检索 → ParentDoc Retriever（细检索粗返回）
    3. 保留 Stage 1 的混合检索 + Rerank + 改写能力
    """
    import torch
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever, ParentDocumentRetriever
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import OllamaLLM
    from langchain_core.stores import InMemoryStore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import jieba

    print("\n[初始化] Stage 3 Pipeline (语义切块 + ParentDoc + Rerank)...")

    # 1. 加载原始文档
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    raw_docs = loader.load()

    # 2. 语义切块（替代 RecursiveTextSplitter）
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
    )
    semantic_chunks = semantic_splitter.split_documents(raw_docs)
    print(f"  语义切块: {len(semantic_chunks)} 个块")

    # 3. 用父块切分器构建 Parent-Document Retriever
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        add_start_index=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        add_start_index=True,
    )

    vectorstore = Chroma(
        collection_name="stage3_parent_doc",
        embedding_function=embeddings,
    )
    store = InMemoryStore()

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    parent_retriever.add_documents(semantic_chunks)

    # 4. 构建 BM25（用父块做索引，关键词匹配更充分）
    parent_chunks = parent_splitter.split_documents(semantic_chunks)
    print(f"  父块: {len(parent_chunks)} 个")

    def chinese_tokenizer(text: str) -> list[str]:
        return list(jieba.cut(text))

    bm25 = BM25Retriever.from_documents(parent_chunks, preprocess_func=chinese_tokenizer)
    bm25.k = HYBRID_K

    # 5. Ensemble: BM25 + ParentDoc 混合
    ensemble = EnsembleRetriever(
        retrievers=[bm25, parent_retriever],
        weights=[0.4, 0.6],
    )

    # 6. Reranker
    print(f"  加载 Reranker: {RERANK_MODEL} ...")
    rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
    rerank_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rerank_model.to(device)
    print(f"    -> device={device}")

    rewrite_llm = OllamaLLM(model=ANSWER_LLM_MODEL, temperature=0)

    def rerank_docs(query, candidates):
        if not candidates:
            return [], []
        # 去重（ParentDoc + BM25 可能返回相同的父块）
        seen = set()
        unique_candidates = []
        for doc in candidates:
            fp = doc.page_content[:80]
            if fp not in seen:
                seen.add(fp)
                unique_candidates.append(doc)
        candidates = unique_candidates

        pairs = [[query, doc.page_content] for doc in candidates]
        with torch.no_grad():
            inputs = rerank_tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            scores = rerank_model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
        sorted_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_pairs[:TOP_N]], [p[1] for p in sorted_pairs[:TOP_N]]

    def rewrite_loop_search(query):
        candidates = ensemble.invoke(query)
        results, scores = rerank_docs(query, candidates)

        if results and scores and scores[0] >= 0.0:
            return results, scores

        rewrite_prompt = f"""你是一个检索专家。用户的原始问题是: {query}
检索系统未能找到相关文档。请将这个问题改写得更具体、更贴近文档中的术语。
只输出改写后的问题,不要加任何解释。"""

        for _attempt in range(2):
            try:
                rewritten = rewrite_llm.invoke(rewrite_prompt).strip()
                candidates = ensemble.invoke(rewritten)
                results, scores = rerank_docs(rewritten, candidates)
                if results and scores and scores[0] >= 0.0:
                    return results, scores
            except Exception:
                pass
        return results, scores

    return rewrite_loop_search


# ============== 6. 主入口 ==============

def run_evaluation(mode: str):
    """运行评估"""
    from langchain_ollama import OllamaLLM

    eval_samples = load_eval_dataset()
    if mode != "compare":
        eval_samples = eval_samples[:10]
        print(f"  (快速模式: 使用前 {len(eval_samples)} 条)")

    print(f"\n[初始化] 答案生成 LLM: {ANSWER_LLM_MODEL}")
    answer_llm = OllamaLLM(model=ANSWER_LLM_MODEL, temperature=0)

    results = {}

    if mode in ("baseline", "compare"):
        pipeline = create_baseline_pipeline()
        print(f"\n{'='*60}")
        print(f"  Baseline: 纯向量检索 (Naive RAG)")
        print(f"{'='*60}")
        ragas_samples = build_ragas_samples(eval_samples, pipeline, answer_llm)
        results["baseline"] = run_ragas_eval(ragas_samples, "Baseline (Naive RAG)")

    if mode in ("improved", "compare"):
        pipeline = create_improved_pipeline()
        print(f"\n{'='*60}")
        print(f"  Improved: Stage 1 完整流水线")
        print(f"{'='*60}")
        ragas_samples = build_ragas_samples(eval_samples, pipeline, answer_llm)
        results["improved"] = run_ragas_eval(ragas_samples, "Improved (Stage 1)")

    if mode in ("stage3", "compare"):
        pipeline = create_stage3_pipeline()
        print(f"\n{'='*60}")
        print(f"  Stage 3: 语义切块 + ParentDoc + Rerank")
        print(f"{'='*60}")
        ragas_samples = build_ragas_samples(eval_samples, pipeline, answer_llm)
        results["stage3"] = run_ragas_eval(ragas_samples, "Stage 3 (Semantic + ParentDoc)")

    # 输出结果
    print(f"\n{'='*60}")
    print("  RAGAS 评估结果")
    print(f"{'='*60}")

    metric_names = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

    for key, result in results.items():
        s = result["scores"]
        print(f"\n[{result['label']}] 耗时={result['elapsed_s']}s")
        for m in metric_names:
            val = s.get(m)
            print(f"  {m:25s}: {val:.4f}" if val is not None else f"  {m:25s}: N/A")

    # 对比
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("  对比")
        print(f"{'='*60}")
        keys = list(results.keys())
        # 两两对比
        for idx_a in range(len(keys)):
            for idx_b in range(idx_a + 1, len(keys)):
                ka, kb = keys[idx_a], keys[idx_b]
                print(f"\n  {results[ka]['label']} -> {results[kb]['label']}")
                sa = results[ka]["scores"]
                sb = results[kb]["scores"]
                for m in metric_names:
                    va = sa.get(m)
                    vb = sb.get(m)
                    if va is not None and vb is not None:
                        delta = vb - va
                        sign = "+" if delta >= 0 else ""
                        print(f"    {m:25s}: {va:.4f} -> {vb:.4f}  ({sign}{delta:.4f})")

    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    result_path = f"reports/ragas_{mode}_{timestamp}.json"

    serializable = {}
    for key, result in results.items():
        serializable[key] = {
            "label": result["label"],
            "scores": result["scores"],
            "elapsed_s": result["elapsed_s"],
        }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 结果已写入 {result_path}")

    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    valid_modes = ("baseline", "improved", "stage3", "compare")
    if mode not in valid_modes:
        print(f"用法: python eval.py [{'|'.join(valid_modes)}]")
        sys.exit(1)

    print("=" * 60)
    print(f"RAGAS 评估 Pipeline - 模式: {mode}")
    print("=" * 60)
    print(f"Judge: {JUDGE_MODEL} @ {JUDGE_BASE_URL}")
    print(f"Answer LLM: {ANSWER_LLM_MODEL} (Ollama)")

    run_evaluation(mode)
