import asyncio
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import yaml
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall

# Adjust if your eval script sits in a subfolder (e.g., from data.app import TelecomRAGService)
from app import TelecomRAGService


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    resolved_path = Path(config_path) if config_path else (Path(__file__).resolve().parent / "config.yaml")
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found at: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


confs = load_config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rag_eval")


@dataclass
class GoldenTestCase:
    case_id: str
    archetype: str
    question: str
    ground_truth_answer: str


@dataclass
class PipelineTrace:
    answer: str
    contexts: list[str]
    retrieval_ms: float
    generation_ms: float
    total_ms: float


def build_matching_embeddings(rag_service: TelecomRAGService) -> HuggingFaceEmbeddings:
    """Uses the same embedding model configured in TelecomRAGService / config.yaml."""
    embed_model = rag_service.confs.get("EMBED_MODEL") or confs.get("EMBED_MODEL")
    if not embed_model:
        raise KeyError("EMBED_MODEL not found in TelecomRAGService config or config.yaml")

    return HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )


async def run_live_rag_pipeline(
    rag_service: TelecomRAGService,
    query: str,
    retrieval_timeout_s: float = 5.0,
    generation_timeout_s: float = 15.0,
) -> PipelineTrace:
    t0 = time.perf_counter()

    t_ret = time.perf_counter()
    retrieved_docs = await asyncio.wait_for(
        asyncio.to_thread(rag_service.retrieve, query), timeout=retrieval_timeout_s
    )
    contexts = [doc.page_content for doc in retrieved_docs]
    retrieval_ms = (time.perf_counter() - t_ret) * 1000

    t_gen = time.perf_counter()
    answer = await asyncio.wait_for(
        asyncio.to_thread(rag_service.ask, query), timeout=generation_timeout_s
    )
    generation_ms = (time.perf_counter() - t_gen) * 1000

    total_ms = (time.perf_counter() - t0) * 1000
    return PipelineTrace(
        answer=answer,
        contexts=contexts,
        retrieval_ms=round(retrieval_ms, 2),
        generation_ms=round(generation_ms, 2),
        total_ms=round(total_ms, 2),
    )


def load_golden_dataset() -> list[GoldenTestCase]:
    return [
        GoldenTestCase(
            case_id="TC-001",
            archetype="Simple Factoid",
            question="How do I activate international roaming?",
            ground_truth_answer=(
                "To activate international roaming, log in to the MyTelecom app or portal, "
                "go to **Plan & Services > International Roaming**, and toggle it on at least "
                "24 hours before you travel. Pick a roaming bundle to avoid per-MB charges. "
                "Roaming will automatically turn off when you return home. If you prefer, you "
                "can also call 611 to enable it."
            ),
        ),
        GoldenTestCase(
            case_id="TC-002",
            archetype="Multi-Hop",
            question="Why is my mobile internet so slow?",
            ground_truth_answer=(
                "Slow speeds are usually caused by network congestion in your area, being in a "
                "low-signal zone, or reaching your high-speed data cap. Try toggling airplane mode "
                "off and on to force a network reconnect. If your high-speed data is exhausted your "
                "speed is throttled to 512 kbps."
            ),
        ),
        GoldenTestCase(
            case_id="TC-003",
            archetype="Out-of-Scope",
            question="What is our company's parental leave policy in Germany?",
            ground_truth_answer="The provided context does not contain information about parental leave in Germany.",
        ),
        GoldenTestCase(
            case_id="TC-004",
            archetype="Simple Factoid",
            question="Can I keep my number when switching to your network?",
            ground_truth_answer=(
                "Yes. Number porting is free and usually completes within 2 hours on working days. "
                "Start the port by calling 611 or visiting a store. Keep your old SIM active until "
                "you receive an SMS confirming the port is complete."
            ),
        ),
    ]


async def score_single_metric(
    metric,
    sample: SingleTurnSample,
    metric_semaphore: asyncio.Semaphore,
    metric_timeout_s: float = 30.0,
) -> float | None:
    try:
        async with metric_semaphore:
            score = await asyncio.wait_for(
                metric.single_turn_ascore(sample), timeout=metric_timeout_s
            )
            val = float(score)
            return None if pd.isna(val) else round(val, 4)
    except asyncio.TimeoutError:
        logger.warning("Metric '%s' timed out after %.1fs", metric.name, metric_timeout_s)
        return None
    except Exception as exc:
        logger.warning("Metric '%s' failed with error: %s", metric.name, exc)
        return None


async def evaluate_single_case(
    case: GoldenTestCase,
    rag_service: TelecomRAGService,
    metrics: list,
    case_semaphore: asyncio.Semaphore,
    metric_semaphore: asyncio.Semaphore,
) -> dict:
    base_row: dict[str, Any] = {
        "case_id": case.case_id,
        "archetype": case.archetype,
        "question": case.question,
        "status": "ok",
        "error": "",
        "response": "",
        "retrieved_contexts": "",
        "retrieval_ms": None,
        "generation_ms": None,
        "total_ms": None,
    }
    for m in metrics:
        base_row[m.name] = None

    try:
        async with case_semaphore:
            trace = await run_live_rag_pipeline(rag_service, case.question)

        base_row.update(
            {
                "retrieval_ms": trace.retrieval_ms,
                "generation_ms": trace.generation_ms,
                "total_ms": trace.total_ms,
                "response": trace.answer,
                "retrieved_contexts": "\n---\n".join(trace.contexts),
            }
        )

        sample = SingleTurnSample(
            user_input=case.question,
            retrieved_contexts=trace.contexts,
            response=trace.answer,
            reference=case.ground_truth_answer,
        )

        metric_tasks = [
            score_single_metric(m, sample, metric_semaphore) for m in metrics
        ]
        metric_scores = await asyncio.gather(*metric_tasks)

        for m, score in zip(metrics, metric_scores):
            base_row[m.name] = score

        if any(s is None for s in metric_scores):
            base_row["status"] = "partial"

        return base_row

    except asyncio.TimeoutError:
        logger.error("Case %s (%s) timed out during live RAG execution", case.case_id, case.archetype)
        base_row["status"] = "failed"
        base_row["error"] = "rag_pipeline_timeout"
        return base_row
    except Exception as exc:
        logger.exception("Case %s (%s) failed: %s", case.case_id, case.archetype, exc)
        base_row["status"] = "failed"
        base_row["error"] = f"{type(exc).__name__}: {exc}"
        return base_row


async def evaluate_live_pipeline(
    test_cases: list[GoldenTestCase], rag_service: TelecomRAGService
) -> pd.DataFrame:
    openai_key = os.environ.get("OPENAI_API_KEY") or confs.get("OPENAI_API_KEY")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    else:
        raise EnvironmentError("Set OPENAI_API_KEY in environment or config.yaml before running evaluation.")

    if not rag_service.is_ready:
        raise RuntimeError("Initialize the RAG service before running evaluation.")

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=3, request_timeout=20.0)
    )
    # Same HuggingFace embedding model as TelecomRAGService
    hf_embeddings = build_matching_embeddings(rag_service)
    judge_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    metrics = [
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
    ]

    case_semaphore = asyncio.Semaphore(3)
    metric_semaphore = asyncio.Semaphore(6)

    tasks = [
        evaluate_single_case(tc, rag_service, metrics, case_semaphore, metric_semaphore)
        for tc in test_cases
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rows: list[dict[str, Any]] = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            tc = test_cases[idx]
            logger.error("Unexpected task crash on %s: %s", tc.case_id, res)
            rows.append(
                {
                    "case_id": tc.case_id,
                    "archetype": tc.archetype,
                    "question": tc.question,
                    "status": "failed",
                    "error": repr(res),
                }
            )
        else:
            rows.append(res)

    return pd.DataFrame(rows)


def print_and_check_gates(df: pd.DataFrame, max_error_rate: float = 0.05) -> None:
    from tabulate import tabulate
    print(tabulate(df, headers='keys', tablefmt='psql'))
    total_cases = len(df)
    failed_cases = int((df["status"] == "failed").sum())
    partial_cases = int((df["status"] == "partial").sum())
    error_rate = (failed_cases / total_cases) if total_cases else 1.0

    logger.info(
        "Run summary: total=%d, ok=%d, partial=%d, failed=%d (error_rate=%.1f%%)",
        total_cases,
        int((df["status"] == "ok").sum()),
        partial_cases,
        failed_cases,
        error_rate * 100,
    )

    df.to_csv("live_rag_eval_matrix.csv", index=False)

    valid_df = df[df["status"] != "failed"].copy()
    if valid_df.empty:
        logger.error("All test cases failed; failing CI gate.")
        sys.exit(1)

    numeric_cols = [
        c
        for c in [
            "llm_context_precision_with_reference",
            "llm_context_recall",
            "faithfulness",
            "answer_relevancy",
            "retrieval_ms",
            "generation_ms",
            "total_ms",
        ]
        if c in valid_df.columns
    ]

    matrix = valid_df.groupby("archetype", sort=False)[numeric_cols].mean().round(3)
    overall = valid_df[numeric_cols].mean().round(3).to_frame(name="VALID_AVG").T

    p95_latency = (
        float(valid_df["total_ms"].dropna().quantile(0.95))
        if "total_ms" in valid_df.columns and not valid_df["total_ms"].dropna().empty
        else float("nan")
    )

    print("\n=== Live RAG Evaluation Matrix (Valid Cases) ===")
    print(pd.concat([matrix, overall]).to_string())
    print(f"\nP95 End-to-End Latency: {p95_latency:.2f} ms")

    failed_rows = df[df["status"] == "failed"]
    if not failed_rows.empty:
        print("\n=== Failed Cases ===")
        print(failed_rows[["case_id", "archetype", "error"]].to_string(index=False))

    failures: list[str] = []
    if error_rate > max_error_rate:
        failures.append(f"error_rate ({error_rate:.1%} > {max_error_rate:.1%})")

    faith_mean = valid_df["faithfulness"].mean() if "faithfulness" in valid_df else float("nan")
    recall_mean = (
        valid_df["llm_context_recall"].mean() if "llm_context_recall" in valid_df else float("nan")
    )

    if pd.isna(faith_mean) or faith_mean < 0.85:
        failures.append(f"faithfulness ({faith_mean:.3f} < 0.85)")
    if pd.isna(recall_mean) or recall_mean < 0.80:
        failures.append(f"llm_context_recall ({recall_mean:.3f} < 0.80)")
    if pd.isna(p95_latency) or p95_latency > 2500.0:
        failures.append(f"p95_total_ms ({p95_latency:.1f}ms > 2500.0ms)")

    if failures:
        print("\n[CI GATE FAILED]:", " | ".join(failures))
        sys.exit(1)

    print("\n[CI GATE PASSED]: Quality, latency, and error-rate thresholds met.")


async def main() -> None:
    test_cases = load_golden_dataset()
    rag_service = TelecomRAGService()
    rag_service.initialize()
    if not rag_service.is_ready:
        raise RuntimeError("RAG service initialization failed; evaluation cannot continue.")
    results_df = await evaluate_live_pipeline(test_cases, rag_service)
    print_and_check_gates(results_df)


if __name__ == "__main__":
    asyncio.run(main())