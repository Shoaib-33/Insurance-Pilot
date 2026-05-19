import json
import re
from typing import Any

from app.core.config import settings
from app.services.groq_llm import GroqLLM


class SelfRAG:
    def __init__(self) -> None:
        self.llm = GroqLLM()

    def grade_retrieval_need(self, query: str) -> dict[str, Any]:
        normalized = re.sub(r"\s+", " ", query.lower()).strip(" ?!.")
        normalized = re.sub(r"^\d+[\).\-\s]+", "", normalized).strip(" ?!.")
        simple_markers = {"hello", "hi", "hey", "help", "what can you do"}
        fallback = {
            "should_retrieve": normalized not in simple_markers,
            "retrieve": normalized not in simple_markers,
            "intent": "smalltalk" if normalized in simple_markers else "out_of_domain",
            "risk_level": "low" if normalized in simple_markers else "medium",
        }
        result = self.llm.invoke_json(
            system=(
                "You are the planner for an insurance RAG assistant. Classify the user's query and "
                "decide whether retrieval from the insurance knowledge base is needed.\n\n"
                "Return JSON with exactly these keys:\n"
                "- should_retrieve: boolean\n"
                "- retrieve: boolean, same value as should_retrieve\n"
                "- intent: one of smalltalk, general_insurance_concept, claim_scenario, out_of_domain\n"
                "- risk_level: one of low, medium, high\n\n"
                "Use general_insurance_concept for educational questions about insurance terms, "
                "regulation, compliance, procedures, definitions, or how insurance works. These "
                "questions should retrieve if they are insurance-related.\n"
                "Use claim_scenario when the user describes an event, loss, damage, theft, injury, "
                "death, bill, repair, approval, denial, coverage, or asks whether insurance will pay. "
                "These questions should retrieve.\n"
                "Use smalltalk only for greetings or capability questions. These usually do not retrieve.\n"
                "Use out_of_domain for questions that are not about insurance, insurance claims, "
                "coverage, policies, documents, claim procedures, regulations, or this assistant's "
                "insurance capability. Medical treatment, medicine dosage, birthday wishes, general "
                "chitchat beyond a greeting, homework, coding, travel planning, and unrelated advice "
                "are out_of_domain and should not retrieve.\n"
                "High risk means coverage decisions, denial, settlement, legal, fraud, death, injury, "
                "large loss, regulatory complaint, or money."
            ),
            user=f"Query: {query}",
            fallback=fallback,
        )
        intent = str(result.get("intent", fallback["intent"]))
        if intent not in {"smalltalk", "general_insurance_concept", "claim_scenario", "out_of_domain"}:
            intent = fallback["intent"]
        should_retrieve = bool(result.get("should_retrieve", fallback["should_retrieve"]))
        if intent in {"general_insurance_concept", "claim_scenario"}:
            should_retrieve = True
        if intent in {"smalltalk", "out_of_domain"}:
            should_retrieve = False
        risk_level = str(result.get("risk_level", fallback["risk_level"]))
        if risk_level not in {"low", "medium", "high"}:
            risk_level = fallback["risk_level"]
        return {
            "should_retrieve": should_retrieve,
            "retrieve": should_retrieve,
            "intent": intent,
            "risk_level": risk_level,
        }

    def critique(
        self,
        query: str,
        answer: str,
        sources: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        if not sources:
            return {
                "passed": False,
                "retrieve": True,
                "isrel": False,
                "issup": False,
                "isuse": False,
                "confidence": 0.35,
                "relevance_score": 0.0,
                "faithfulness_score": 0.0,
                "evidence_score": 0.0,
                "needs_rewrite": iteration == 0,
                "rewrite_query": query,
                "issues": ["No retrieved evidence was available."],
            }

        if settings.low_latency_mode:
            return self._heuristic_critique(answer, sources, iteration)

        evidence = "\n\n".join(
            f"[{i + 1}] {src.get('source_name', 'source')} :: {src.get('text', '')[:900]}"
            for i, src in enumerate(sources)
        )
        fallback = self._heuristic_critique(answer, sources, iteration)
        result = self.llm.invoke_json(
            system=(
                "You are a Self-RAG evaluator for an insurance claims assistant. Return JSON with "
                "the classic Self-RAG labels: retrieve, isrel, issup, isuse. Definitions: retrieve "
                "means external evidence was needed; ISREL means retrieved passages are relevant; "
                "ISSUP means the generated answer is supported by those passages; ISUSE means the "
                "overall response is useful for the user's claim scenario. Also return passed, "
                "confidence, relevance_score, faithfulness_score, evidence_score, needs_rewrite, "
                "rewrite_query, and issues."
            ),
            user=f"Query:\n{query}\n\nDraft answer:\n{answer}\n\nEvidence:\n{evidence}",
            fallback=fallback,
        )
        return {
            "passed": bool(result.get("passed", False)),
            "retrieve": bool(result.get("retrieve", True)),
            "isrel": bool(result.get("isrel", False)),
            "issup": bool(result.get("issup", False)),
            "isuse": bool(result.get("isuse", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "relevance_score": float(result.get("relevance_score", 0.0)),
            "faithfulness_score": float(result.get("faithfulness_score", 0.0)),
            "evidence_score": float(result.get("evidence_score", 0.0)),
            "needs_rewrite": bool(result.get("needs_rewrite", False)),
            "rewrite_query": result.get("rewrite_query") or query,
            "issues": result.get("issues", []),
        }

    def _heuristic_critique(self, answer: str, sources: list[dict[str, Any]], iteration: int) -> dict[str, Any]:
        source_count = len(sources)
        has_answer = len(answer.strip()) > 40
        answer_lower = answer.lower()
        has_decision = "decision:" in answer_lower
        has_missing = "missing evidence:" in answer_lower
        has_action = "recommended action" in answer_lower or "recommended tool" in answer_lower
        has_citation = "[source" in answer_lower
        query_terms = set()
        source_terms = set()
        for source in sources:
            source_terms.update(re.findall(r"[a-zA-Z0-9_]+", source.get("text", "").lower()))
        isrel = source_count > 0 and bool(source_terms)
        issup = has_citation and source_count > 0
        isuse = has_answer and has_decision and has_missing and has_action
        confidence = min(
            0.95,
            0.35
            + source_count * 0.07
            + (0.15 if has_answer else 0)
            + (0.15 if isrel else 0)
            + (0.15 if issup else 0)
            + (0.15 if isuse else 0),
        )
        passed = isrel and issup and isuse and (confidence >= 0.68 or iteration > 0)
        issues = []
        if not isrel:
            issues.append("ISREL failed: retrieved passages appear weak or missing.")
        if not issup:
            issues.append("ISSUP failed: answer lacks clear support citation.")
        if not isuse:
            issues.append("ISUSE failed: answer is missing decision, missing evidence, or action structure.")
        return {
            "passed": passed,
            "retrieve": True,
            "isrel": isrel,
            "issup": issup,
            "isuse": isuse,
            "confidence": confidence,
            "relevance_score": 0.9 if isrel else 0.25,
            "faithfulness_score": 0.85 if issup else 0.35,
            "evidence_score": min(0.9, 0.35 + source_count * 0.1),
            "needs_rewrite": not passed and iteration == 0,
            "rewrite_query": None,
            "issues": issues,
        }


def safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True)
