import os
import time
from pathlib import Path
import pandas as pd
import yaml
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall

from app import TelecomRAGService

load_dotenv()

# 1. LOAD CONFIGURATION & SET API KEY
def setup_environment():
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        confs = yaml.safe_load(f) or {}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment or config.yaml")
    
    os.environ["OPENAI_API_KEY"] = api_key
    return confs


# 2. DEFINE TEST CASES (GOLDEN DATASET)
def get_test_cases():
    return [
        {
            "id": "TC-001",
            "type": "Simple Factoid",
            "question": "How do I activate international roaming?",
            "ground_truth": (
                "To activate international roaming, log in to the MyTelecom app or portal, "
                "go to **Plan & Services > International Roaming**, and toggle it on at least "
                "24 hours before you travel. Pick a roaming bundle to avoid per-MB charges. "
                "Roaming will automatically turn off when you return home. If you prefer, you "
                "can also call 611 to enable it."
            ),
        },
        {
            "id": "TC-002",
            "type": "Multi-Hop",
            "question": "Why is my mobile internet so slow?",
            "ground_truth": (
                "Slow speeds are usually caused by network congestion in your area, being in a "
                "low-signal zone, or reaching your high-speed data cap. Try toggling airplane mode "
                "off and on to force a network reconnect. If your high-speed data is exhausted your "
                "speed is throttled to 512 kbps."
            ),
        },
        {
            "id": "TC-003",
            "type": "Out-of-Scope",
            "question": "What is our company's parental leave policy in Germany?",
            "ground_truth": "The provided context does not contain information about parental leave in Germany.",
        },
        {
            "id": "TC-004",
            "type": "Simple Factoid",
            "question": "Can I keep my number when switching to your network?",
            "ground_truth": (
                "Yes. Number porting is free and usually completes within 2 hours on working days. "
                "Start the port by calling 611 or visiting a store. Keep your old SIM active until "
                "you receive an SMS confirming the port is complete."
            ),
        },
    ]


# 3. INITIALIZE RAGAS METRICS (THE "JUDGE")
def setup_ragas_metrics(embed_model_name):
    print(f"\n[Setup] Initializing Judge LLM (gpt-4o-mini) and Embeddings ({embed_model_name})...")
    
    # LLM used by Ragas to grade the answers
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    
    # Embeddings used by Ragas (matching your RAG app's embedding model)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    judge_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # The 4 core Ragas metrics
    return [
        LLMContextPrecisionWithReference(llm=judge_llm), # Did retrieval rank relevant chunks at the top?
        LLMContextRecall(llm=judge_llm),                 # Did retrieval find all info needed for ground truth?
        Faithfulness(llm=judge_llm),                     # Is the answer strictly based on retrieved context (no hallucinations)?
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings), # Does the answer actually address the question?
    ]


# 4. MAIN EVALUATION LOOP
def main():
    print("=== STARTING RAG EVALUATION ===")
    confs = setup_environment()

    # Initialize your RAG pipeline
    print("\n[Step 1] Initializing TelecomRAGService...")
    rag_service = TelecomRAGService()
    rag_service.initialize()
    if not rag_service.is_ready:
        raise RuntimeError("RAG service failed to initialize!")

    embed_model = rag_service.confs.get("EMBED_MODEL") or confs.get("EMBED_MODEL")
    metrics = setup_ragas_metrics(embed_model)
    test_cases = get_test_cases()

    results = []

    # Loop through each test case one by one
    for case in test_cases:
        print("\n" + "=" * 60)
        print(f"Running {case['id']} ({case['type']})")
        print(f"Question: {case['question']}")
        print("=" * 60)

        # A. Run Retrieval & Measure Time
        t0 = time.perf_counter()
        retrieved_docs = rag_service.retrieve(case["question"])
        contexts = [doc.page_content for doc in retrieved_docs]
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)

        print(f"\n-> Retrieved {len(contexts)} context chunks in {retrieval_ms} ms.")
        for i, ctx in enumerate(contexts, 1):
            print(f"   [Chunk {i}]: {ctx[:100]}...") # Print first 100 chars of each chunk

        # B. Run Generation & Measure Time
        t1 = time.perf_counter()
        answer = rag_service.ask(case["question"])
        generation_ms = round((time.perf_counter() - t1) * 1000, 2)

        print(f"\n-> Generated Answer in {generation_ms} ms:")
        print(f"   \"{answer}\"")
        print(f"\n-> Expected Ground Truth:")
        print(f"   \"{case['ground_truth']}\"")

        # C. Package data into a Ragas SingleTurnSample
        sample = SingleTurnSample(
            user_input=case["question"],
            retrieved_contexts=contexts,
            response=answer,
            reference=case["ground_truth"],
        )

        # D. Score the sample using each Ragas metric
        print("\n-> Calculating Ragas Scores...")
        row = {
            "case_id": case["id"],
            "archetype": case["type"],
            "question": case["question"],
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": round(retrieval_ms + generation_ms, 2),
        }

        import traceback # Add this at the top of your file

    # Replace the scoring loop inside main() with this:
        for metric in metrics:
            try:
                # Use .single_turn_score() for SingleTurnSample objects
                score = metric.single_turn_score(sample)
                
                if pd.isna(score):
                    print(f"   [Warning] {metric.name} returned NaN (check OpenAI API/JSON output)")
                    score = None
                else:
                    score = round(float(score), 4)
                    
            except Exception as e:
                print(f"   [Error] {metric.name} failed: {e}")
                traceback.print_exc() # Prints the exact reason it failed
                score = None

            row[metric.name] = score
            print(f"   * {metric.name}: {score}")
        results.append(row)
    df = pd.DataFrame(results)


    # 5. DISPLAY SUMMARY & SAVE TO CSV
    print("\n" + "=" * 60)
    print("=== FINAL EVALUATION SUMMARY ===")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    print("\n=== AVERAGE SCORES ACROSS ALL CASES ===")
    metric_cols = [m.name for m in metrics] + ["retrieval_ms", "generation_ms", "total_ms"]
    print(df[metric_cols].mean().round(3).to_string())

    df.to_csv("simple_rag_eval_results.csv", index=False)
    print("\nSaved detailed results to 'simple_rag_eval_results.csv'")


if __name__ == "__main__":
    main()
