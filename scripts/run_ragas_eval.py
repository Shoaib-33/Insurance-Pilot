import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.core.config import settings
from app.db.sqlite import init_db
from app.rag.embeddings import get_embedding_model
from app.rag.graph import ClaimsRAGGraph
from app.rag.ingestion import DocumentIngestionService
from app.rag.qdrant_store import QdrantVectorStore


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def default_reference(case: dict[str, Any]) -> str:
    if case.get("reference"):
        return str(case["reference"])
    decision = case["expected_decision"]
    return (
        f"Decision: {decision}. The answer should use the retrieved insurance claim "
        "guidance to explain the coverage triage, identify missing evidence, and "
        "recommend the next action without inventing unsupported policy terms."
    )


def build_eval_rows(dataset_path: Path, user_id: str, limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    init_db()
    QdrantVectorStore().ensure_collections()
    DocumentIngestionService().ingest_pdf_directory()
    graph = ClaimsRAGGraph()

    cases = load_jsonl(dataset_path)
    if limit:
        cases = cases[:limit]

    ragas_rows = []
    run_rows = []
    for case in cases:
        started = time.perf_counter()
        state = graph.run(case["query"], user_id=user_id, use_cache=False)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        sources = state.get("reranked_sources") or state.get("sources", [])
        contexts = [str(source.get("text", "")) for source in sources if source.get("text")]

        ragas_rows.append(
            {
                "user_input": case["query"],
                "response": state.get("answer", ""),
                "retrieved_contexts": contexts,
                "reference": default_reference(case),
            }
        )
        run_rows.append(
            {
                "id": case["id"],
                "expected_decision": case["expected_decision"],
                "sources": len(contexts),
                "latency_ms": latency_ms,
            }
        )
    return ragas_rows, run_rows


async def run_ragas(dataset_path: Path, user_id: str, limit: int | None) -> dict[str, Any]:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for RAGAS LLM-judge metrics.")

    ragas_rows, run_rows = build_eval_rows(dataset_path, user_id, limit)
    dataset = Dataset.from_list(ragas_rows)

    judge_llm = ChatGroq(
        model=settings.groq_model,
        temperature=0,
        max_retries=2,
        api_key=settings.groq_api_key,
    )
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embedding_model().model)

    answer_relevancy.strictness = 1
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(timeout=180, max_workers=2, max_retries=2),
    )

    scores = result.to_pandas().to_dict(orient="records")
    rows = []
    for run_row, score_row in zip(run_rows, scores, strict=False):
        rows.append({**run_row, "ragas": score_row})

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = {
        "total": len(rows),
        "metrics": {},
        "results": rows,
    }
    for metric in metric_names:
        values = [
            float(row["ragas"][metric])
            for row in rows
            if row["ragas"].get(metric) is not None and str(row["ragas"][metric]).lower() != "nan"
        ]
        summary["metrics"][metric] = round(sum(values) / len(values), 3) if values else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/golden_claim_scenarios.jsonl")
    parser.add_argument("--user-id", default="ragas_eval_user")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    summary = asyncio.run(run_ragas(Path(args.dataset), args.user_id, args.limit))
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
