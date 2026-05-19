import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.sqlite import init_db
from app.rag.graph import ClaimsRAGGraph
from app.rag.ingestion import DocumentIngestionService
from app.rag.qdrant_store import QdrantVectorStore


DECISION_RE = re.compile(r"Decision:\s*(Likely covered|Likely not covered|Needs human review)", re.I)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def extract_decision(answer: str) -> str:
    match = DECISION_RE.search(answer)
    if not match:
        return "Unknown"
    return match.group(1).capitalize().replace("Not", "not")


def accepted_decisions(case: dict[str, Any]) -> set[str]:
    values = case.get("acceptable_decisions") or [case["expected_decision"]]
    return {str(v).lower() for v in values}


def evaluate(dataset_path: Path, user_id: str, limit: int | None = None) -> dict[str, Any]:
    init_db()
    QdrantVectorStore().ensure_collections()
    DocumentIngestionService().ingest_pdf_directory()
    graph = ClaimsRAGGraph()

    cases = load_jsonl(dataset_path)
    if limit:
        cases = cases[:limit]
    results = []
    for case in cases:
        started = time.perf_counter()
        state = graph.run(case["query"], user_id=user_id, use_cache=False)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        answer = state.get("answer", "")
        decision = extract_decision(answer)
        sources = state.get("reranked_sources") or state.get("sources", [])
        critique = state.get("self_rag", {})
        decision_ok = decision.lower() in accepted_decisions(case)
        sources_ok = bool(sources) if case.get("must_have_sources", True) else True
        self_rag_ok = bool(critique.get("isrel")) and bool(critique.get("issup")) and bool(critique.get("isuse"))
        passed = decision_ok and sources_ok and self_rag_ok
        results.append(
            {
                "id": case["id"],
                "expected": case["expected_decision"],
                "decision": decision,
                "decision_ok": decision_ok,
                "sources": len(sources),
                "sources_ok": sources_ok,
                "self_rag": {
                    "ISREL": critique.get("isrel"),
                    "ISSUP": critique.get("issup"),
                    "ISUSE": critique.get("isuse"),
                },
                "self_rag_ok": self_rag_ok,
                "latency_ms": latency_ms,
                "passed": passed,
            }
        )

    total = len(results)
    summary = {
        "total": total,
        "passed": sum(1 for r in results if r["passed"]),
        "decision_accuracy": round(sum(1 for r in results if r["decision_ok"]) / total, 3) if total else 0,
        "source_rate": round(sum(1 for r in results if r["sources_ok"]) / total, 3) if total else 0,
        "self_rag_pass_rate": round(sum(1 for r in results if r["self_rag_ok"]) / total, 3) if total else 0,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / total, 2) if total else 0,
        "results": results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/golden_claim_scenarios.jsonl")
    parser.add_argument("--user-id", default="eval_user")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = evaluate(Path(args.dataset), args.user_id, args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
