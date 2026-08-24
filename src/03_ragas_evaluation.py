"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import types
import hashlib
import math
import re
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings


def _install_ragas_vertexai_compatibility() -> None:
    """Provide the legacy VertexAI import expected by RAGAS 0.4.x.

    RAGAS 0.4.3 imports ``ChatVertexAI`` only to identify models that support
    multiple completions.  Recent ``langchain-community`` releases moved that
    integration to a separate package and removed the legacy module.  The lab
    uses our provider factory instead, so a lightweight type-only compatibility
    module keeps RAGAS importable without pinning the rest of LangChain to an
    obsolete release.
    """
    module_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(module_name, fromlist=["ChatVertexAI"])
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    compatibility_module = types.ModuleType(module_name)

    class ChatVertexAI:  # noqa: N801 - preserve the legacy public name
        """Type marker used by RAGAS' multiple-completion check."""

    compatibility_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = compatibility_module


_install_ragas_vertexai_compatibility()

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Chỉ sử dụng thông tin trong context để trả lời "
    "và giữ câu trả lời ngắn gọn, rõ ràng trong 2-4 câu. "
    "Nếu context không đủ thông tin, hãy nói thẳng rằng bạn không biết.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Đọc kỹ context và trả lời có cấu trúc "
    "trong 3-5 câu: nêu kết luận chính, giải thích các facts liên quan, rồi chỉ ra "
    "mức độ chắc chắn nếu phù hợp. Luôn bám sát context và không suy đoán thêm.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
CACHE_DIR = Path(__file__).parent / ".ragas_cache"


class HashingEmbeddings(Embeddings):
    """Dependency-free local embeddings for RAGAS similarity calculations.

    The RAG index itself uses Gemini embeddings.  This evaluator-only fallback
    keeps answer relevancy runnable after the free daily embedding quota is
    exhausted, without sending any additional document content to an API.
    """

    dimensions = 512

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def invoke_with_quota_retry(chain, inputs: dict) -> str:
    """Invoke a chain again after Gemini reports a temporary rate limit."""
    for attempt in range(3):
        try:
            return chain.invoke(inputs)
        except Exception as error:
            message = str(error)
            if attempt == 2 or ("RESOURCE_EXHAUSTED" not in message and "429" not in message):
                raise
            match = re.search(r"retry in ([0-9.]+)s", message, flags=re.IGNORECASE)
            delay = float(match.group(1)) + 1 if match else 61
            print(f"⏳ Gemini generation quota: chờ {delay:.0f}s rồi thử lại...")
            time.sleep(delay)


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=600, chunk_overlap=60)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    contexts = [doc.page_content for doc in docs]

    # Ghép contexts thành một chuỗi để truyền vào {context} của prompt.
    ctx_str = "\n\n".join(contexts)

    answer = invoke_with_quota_retry(prompt | llm | StrOutputParser(), {
        "context": ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy {len(QA_PAIRS)} câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question'][:60]}")

    return results


def load_cached_rag_outputs(prompt_version: str) -> list | None:
    """Return a complete cached prompt run, if one is available."""
    cache_path = CACHE_DIR / f"{prompt_version}_outputs.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(cached) == len(QA_PAIRS):
            return cached
    return None


def load_or_collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """Reuse completed RAG outputs so an evaluation retry does not re-spend quota."""
    cached = load_cached_rag_outputs(prompt_version)
    if cached is not None:
        print(f"♻️  Dùng cache {len(cached)} câu RAG cho prompt {prompt_version}")
        return cached

    if vectorstore is None:
        raise RuntimeError("Cần vectorstore khi chưa có cache RAG")

    results = collect_rag_outputs(vectorstore, prompt_version)
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    # Keep Flash Lite for the RAG pipeline; use a structured-output-capable
    # model only for RAGAS evaluation.
    llm_eval = get_llm(temperature=0, model_override="gemini-3.1-flash-lite")
    emb_eval = HashingEmbeddings()

    # Gemini 3.5 Flash Lite does not allow multiple candidates in one
    # request.  RAGAS still computes answer relevancy with one generated
    # reverse-question per response, which is the supported configuration.
    answer_relevancy.strictness = 1

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        # Keep a small worker pool.  RAGAS' default 16 workers would queue
        # Gemini requests long enough for them to timeout under the limiter.
        run_config=RunConfig(timeout=900, max_workers=4, max_retries=3, max_wait=65),
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        values = [float(v) for v in raw if v is not None]
        scores[key] = float(np.mean(values)) if values else 0.0

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # If the 100 generated answers are cached, evaluation does not need to
    # recreate the FAISS index or spend additional Gemini embedding quota.
    v1_results = load_cached_rag_outputs("v1")
    v2_results = load_cached_rag_outputs("v2")
    if v1_results is not None and v2_results is not None:
        print(f"♻️  Dùng cache {len(v1_results)} câu RAG cho prompt v1")
        print(f"♻️  Dùng cache {len(v2_results)} câu RAG cho prompt v2")
    else:
        vectorstore = setup_vectorstore()
        v1_results = load_or_collect_rag_outputs(vectorstore, "v1")
        v2_results = load_or_collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
