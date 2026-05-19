import json
import re
from typing import Any

from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.db.sqlite import db
from app.guardrails.pii import PIIGuardrails
from app.guardrails.prompt_injection import detect_prompt_injection, strip_unsafe_retrieved_text
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import FlashRankReranker
from app.rag.semantic_cache import SemanticAnswerCache
from app.rag.self_rag import SelfRAG
from app.rag.state import RagState
from app.rag.text import new_id, normalize_text
from app.services.groq_llm import GroqLLM
from app.services.memory import ClaimMemoryService


class ClaimsRAGGraph:
    def __init__(self) -> None:
        self.cache = SemanticAnswerCache()
        self.guardrails = PIIGuardrails()
        self.retriever = HybridRetriever()
        self.reranker = FlashRankReranker()
        self.self_rag = SelfRAG()
        self.llm = GroqLLM()
        self.memory = ClaimMemoryService()
        self.graph = self._build_graph()

    def run(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        user_id: str = "default_user",
        use_cache: bool = True,
    ) -> RagState:
        request_id = new_id("request")
        state: RagState = {
            "request_id": request_id,
            "query": query,
            "user_id": user_id,
            "sanitized_query": query,
            "retrieval_query": query,
            "normalized_query": normalize_text(query),
            "memory_context": "",
            "iteration": 0,
            "cache_hit": False,
            "use_cache": use_cache,
            "trace": [{"node": "start", "query": query, "metadata_filter": metadata_filter or {}}],
        }
        if metadata_filter:
            state["metadata_filter"] = metadata_filter  # type: ignore[typeddict-unknown-key]
        result = self.graph.invoke(state)
        self._save_trace(result)
        return result

    def _build_graph(self):
        workflow = StateGraph(RagState)
        workflow.add_node("planner", self._planner)
        workflow.add_node("semantic_cache", self._semantic_cache)
        workflow.add_node("guardrails", self._guardrails)
        workflow.add_node("load_memory", self._load_memory)
        workflow.add_node("direct_response", self._direct_response)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("rerank", self._rerank)
        workflow.add_node("generate", self._generate)
        workflow.add_node("critique", self._critique)
        workflow.add_node("rewrite", self._rewrite)
        workflow.add_node("finalize", self._finalize)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "semantic_cache")
        workflow.add_conditional_edges(
            "semantic_cache",
            self._route_cache,
            {"hit": "finalize", "miss": "guardrails"},
        )
        workflow.add_conditional_edges(
            "guardrails",
            lambda state: "direct" if state.get("answer") and "override system" in state["answer"] else "memory",
            {"direct": "direct_response", "memory": "load_memory"},
        )
        workflow.add_conditional_edges(
            "load_memory",
            self._route_retrieval,
            {"direct": "direct_response", "out_of_domain": "direct_response", "retrieve": "retrieve"},
        )
        workflow.add_edge("direct_response", "finalize")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "critique")
        workflow.add_conditional_edges(
            "critique",
            self._route_critique,
            {"accept": "finalize", "retry": "rewrite"},
        )
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def _planner(self, state: RagState) -> RagState:
        decision = self.self_rag.grade_retrieval_need(state["query"])
        state["should_retrieve"] = bool(decision.get("should_retrieve", True))
        state["intent"] = str(decision.get("intent", "claims_question"))
        state["risk_level"] = decision.get("risk_level", "medium")  # type: ignore[assignment]
        self._trace(state, "planner", decision)
        return state

    def _load_memory(self, state: RagState) -> RagState:
        memory_context = self.memory.search(
            user_id=state.get("user_id", "default_user"),
            query=state["sanitized_query"],
        )
        state["memory_context"] = memory_context
        self._trace(
            state,
            "load_memory",
            {
                "langmem_available": self.memory.langmem_available,
                "has_memory": memory_context != "No prior memory for this user.",
            },
        )
        return state

    def _semantic_cache(self, state: RagState) -> RagState:
        if not state.get("use_cache", True):
            state["cache_hit"] = False
            self._trace(state, "semantic_cache", {"hit": False, "disabled": True})
            return state
        hit = self.cache.lookup(state["query"])
        if hit:
            state["cache_hit"] = True
            state["answer"] = hit["answer"]
            state["confidence"] = hit["confidence"]
            state["sources"] = hit["sources"]
            has_sources = bool(hit["sources"])
            state["self_rag"] = {
                "passed": True,
                "retrieve": True,
                "isrel": has_sources,
                "issup": has_sources,
                "isuse": bool(hit["answer"]),
                "confidence": hit["confidence"],
                "issues": ["Served from semantic answer cache."],
            }
            self._trace(state, "semantic_cache", {"hit": True, "score": hit["score"]})
            return state
        state["cache_hit"] = False
        self._trace(state, "semantic_cache", {"hit": False})
        return state

    def _guardrails(self, state: RagState) -> RagState:
        pii = self.guardrails.sanitize(state["query"])
        injection = detect_prompt_injection(state["query"])
        state["sanitized_query"] = pii.text
        if injection:
            state["should_retrieve"] = False
            state["answer"] = "I cannot help with requests that try to override system or safety instructions."
            state["confidence"] = 0.99
        self._trace(
            state,
            "guardrails",
            {
                "pii_findings": pii.findings,
                "prompt_injection_findings": injection,
                "langchain_pii_middlewares": len(self.guardrails.langchain_middlewares),
            },
        )
        return state

    def _direct_response(self, state: RagState) -> RagState:
        if state.get("answer"):
            return state
        if state.get("intent") == "out_of_domain":
            state["answer"] = self._scope_message()
            state["confidence"] = 0.98
            state["sources"] = []
            state["self_rag"] = {
                "passed": True,
                "retrieve": False,
                "isrel": False,
                "issup": False,
                "isuse": True,
                "confidence": 0.98,
                "issues": ["Out-of-domain request blocked before retrieval."],
            }
            self._trace(state, "direct_response", {"reason": "out_of_domain"})
            return state
        if state.get("intent") == "smalltalk":
            state["answer"] = self._scope_message()
            state["confidence"] = 0.9
            state["sources"] = []
            state["self_rag"] = {
                "passed": True,
                "retrieve": False,
                "isrel": False,
                "issup": False,
                "isuse": True,
                "confidence": 0.9,
                "issues": ["Smalltalk redirected to insurance scope."],
            }
            self._trace(state, "direct_response", {"reason": "smalltalk"})
            return state
        answer = self.llm.invoke_text(
            system=(
                "You are an insurance claims support copilot. Answer simple non-policy questions "
                "briefly. Do not invent coverage, policy terms, claim outcomes, or payments."
            ),
            user=state["sanitized_query"],
        )
        state["answer"] = answer
        state["confidence"] = 0.72
        state["sources"] = []
        state["self_rag"] = {
            "passed": True,
            "confidence": 0.72,
            "issues": ["Direct response path. Retrieval was not required."],
        }
        self._trace(state, "direct_response", {"confidence": state["confidence"]})
        return state

    def _scope_message(self) -> str:
        return (
            "I can only help with insurance-related claims, coverage, policy terms, claim "
            "documents, and claim procedures. Please ask an insurance claim question."
        )

    def _retrieve(self, state: RagState) -> RagState:
        query = self._prepare_retrieval_query(state)
        metadata_filter = state.get("metadata_filter")  # type: ignore[typeddict-item]
        sources = self.retriever.retrieve(query, metadata_filter=metadata_filter)
        cleaned_sources = []
        for source in sources:
            cleaned_sources.append({**source, "text": strip_unsafe_retrieved_text(source.get("text", ""))})
        state["sources"] = cleaned_sources
        self._trace(state, "retrieve", {"count": len(cleaned_sources), "retrieval_query": query})
        return state

    def _rerank(self, state: RagState) -> RagState:
        reranked = self.reranker.rerank(
            state.get("retrieval_query", state["sanitized_query"]),
            state.get("sources", []),
            top_k=settings.rerank_top_k,
        )
        state["reranked_sources"] = reranked
        self._trace(state, "rerank", {"count": len(reranked)})
        return state

    def _prepare_retrieval_query(self, state: RagState) -> str:
        if not settings.enable_query_rewrite:
            state["retrieval_query"] = state["sanitized_query"]
            return state["retrieval_query"]
        if state.get("retrieval_query") and state.get("retrieval_query") != state.get("query"):
            return state["retrieval_query"]
        result = self.llm.invoke_json(
            system=(
                "You rewrite user insurance questions into concise retrieval queries for a hybrid "
                "BM25 + vector RAG system. Preserve all facts from the user. Do not answer the "
                "question. Add only helpful insurance terminology that improves retrieval, such as "
                "coverage part, exclusion, deductible, endorsement, claim documents, fault, valuation, "
                "or policy limit when relevant.\n\n"
                "Important retrieval discipline:\n"
                "- Keep the rewritten query focused on the user's requested claim issue.\n"
                "- Do not add benefits, services, or subtopics the user did not ask about.\n"
                "- If the user asks about damage to insured property, focus on the coverage for that "
                "damage, required evidence, deductible, and exclusions.\n"
                "- If a term in the user question is vague, rewrite it into precise insurance language, "
                "but do not change the claim type.\n"
                "- Prefer compact keyword-rich wording over a sentence.\n\n"
                "Return JSON only with keys: query, changed, rationale."
            ),
            user=(
                f"Intent: {state.get('intent', 'unknown')}\n"
                f"Original user question:\n{state['sanitized_query']}"
            ),
            fallback={"query": state["sanitized_query"], "changed": False, "rationale": "fallback"},
        )
        rewritten = str(result.get("query") or state["sanitized_query"]).strip()
        if not rewritten:
            rewritten = state["sanitized_query"]
        state["retrieval_query"] = rewritten
        self._trace(
            state,
            "query_rewrite",
            {
                "original_query": state["sanitized_query"],
                "retrieval_query": rewritten,
                "changed": bool(result.get("changed", rewritten != state["sanitized_query"])),
                "rationale": str(result.get("rationale", ""))[:300],
            },
        )
        return rewritten

    def _generate(self, state: RagState) -> RagState:
        sources = state.get("reranked_sources", [])
        llm_sources = sources[: settings.max_sources_to_llm]
        evidence = "\n\n".join(
            (
                f"Source {i + 1}: {src.get('source_name', 'unknown')}\n"
                f"Text: {src.get('text', '')[: settings.max_evidence_chars_per_source]}"
            )
            for i, src in enumerate(llm_sources)
        )
        if state.get("intent") == "general_insurance_concept":
            answer = self.llm.invoke_text(
                system=(
                    "You are an insurance education assistant. The user is asking a general "
                    "insurance concept question, not requesting a claim payment decision. Use only "
                    "the provided evidence. Answer briefly and clearly in plain language. Do not use "
                    "the claim triage structure. Do not say Likely covered, Likely not covered, or "
                    "Needs human review unless the user asks about a claim scenario. Cite sources as "
                    "[Source 1], [Source 2], etc. Do not use outside source names."
                ),
                user=(
                    f"Question:\n{state['sanitized_query']}\n\n"
                    f"Retrieved evidence:\n{evidence}"
                ),
            )
            state["answer"] = self._ensure_source_citation(answer, sources)
            self._trace(state, "generate", {"source_count": len(sources), "mode": "concept_llm"})
            return state
        answer = self._generate_claim_json_answer(state, evidence, sources)
        state["answer"] = self._ensure_source_citation(answer, sources)
        self._trace(state, "generate", {"source_count": len(sources), "mode": "llm"})
        return state

    def _ensure_source_citation(self, answer: str, sources: list[dict[str, Any]]) -> str:
        if not sources or re.search(r"\[Source\s+\d+\]", answer, flags=re.IGNORECASE):
            return answer
        return answer.rstrip() + " [Source 1]"

    def _generate_claim_json_answer(self, state: RagState, evidence: str, sources: list[dict[str, Any]]) -> str:
        result = self.llm.invoke_json(
            system=(
                "You are an insurance claim support AI agent. The user describes a claim scenario. "
                "Use only the retrieved evidence. Do not use outside knowledge. Do not invent final "
                "payment approval, denial, claim status, policy terms, or source names.\n\n"
                "Return JSON only with exactly these keys:\n"
                "- decision: one of Likely covered, Likely not covered, Needs human review\n"
                "- reason: one or two evidence-grounded sentences\n"
                "- missing_evidence: short string listing missing documents/facts or None identified\n"
                "- recommended_action: short string with next step, escalation, or review action\n"
                "- sources: short string with citations like [Source 1], [Source 2]\n\n"
                "Rubric:\n"
                "- Likely covered: use only when evidence directly says this cause of loss or scenario is "
                "normally covered by the relevant coverage part and the user's facts do not leave a major "
                "coverage dependency unresolved.\n"
                "- Likely not covered: use only when evidence directly says this cause of loss is excluded, "
                "not covered by the standard policy, or requires separate coverage that the user says they "
                "do not have.\n"
                "- Needs human review: use when payment or coverage depends on unresolved policy-specific "
                "facts, endorsements, sublimits, deductibles, fault, valuation, contestability, fraud "
                "review, regulatory timing, guaranty fund state limits, settlement amount disputes, or "
                "other claim-file details.\n\n"
                "If evidence is incomplete, still choose the best triage label and explain what is missing. "
                "Allowed citations are only [Source 1], [Source 2], etc."
            ),
            user=(
                f"User memory context:\n{state.get('memory_context', 'No prior memory.')}\n\n"
                f"Claim scenario:\n{state['sanitized_query']}\n\n"
                f"Retrieved evidence:\n{evidence}"
            ),
            fallback={
                "decision": "Needs human review",
                "reason": "The available evidence is not sufficient to make a final coverage triage.",
                "missing_evidence": "Policy-specific details and claim-file documentation.",
                "recommended_action": "Escalate for human review with the retrieved evidence.",
                "sources": "[Source 1]" if sources else "",
            },
        )
        decision = str(result.get("decision", "Needs human review"))
        if decision not in {"Likely covered", "Likely not covered", "Needs human review"}:
            decision = "Needs human review"
        sources_text = str(result.get("sources", "")).strip()
        if sources and not re.search(r"\[Source\s+\d+\]", sources_text, flags=re.IGNORECASE):
            sources_text = "[Source 1]"
        return (
            f"Decision: {decision}\n"
            f"Reason: {str(result.get('reason', '')).strip()}\n"
            f"Missing evidence: {str(result.get('missing_evidence', '')).strip()}\n"
            f"Recommended action: {str(result.get('recommended_action', '')).strip()}\n"
            f"Sources: {sources_text}"
        )

    def _critique(self, state: RagState) -> RagState:
        if state.get("intent") == "general_insurance_concept":
            sources = state.get("reranked_sources", [])
            answer = state.get("answer", "")
            critique = {
                "passed": bool(sources) and bool(answer),
                "retrieve": True,
                "isrel": bool(sources),
                "issup": bool(sources) and "[source" in answer.lower(),
                "isuse": len(answer.strip()) > 20,
                "confidence": 0.84 if sources and answer else 0.45,
                "relevance_score": 0.85 if sources else 0.0,
                "faithfulness_score": 0.8 if sources and "[source" in answer.lower() else 0.35,
                "evidence_score": min(0.9, 0.35 + len(sources) * 0.1),
                "needs_rewrite": False,
                "rewrite_query": None,
                "issues": ["General insurance concept answer with retrieved sources."],
            }
            state["self_rag"] = critique
            state["confidence"] = float(critique["confidence"])
            self._trace(state, "critique", critique)
            return state
        critique = self.self_rag.critique(
            query=state["sanitized_query"],
            answer=state.get("answer", ""),
            sources=state.get("reranked_sources", []),
            iteration=int(state.get("iteration", 0)),
        )
        state["self_rag"] = critique
        state["confidence"] = float(critique.get("confidence", 0.0))
        self._trace(state, "critique", critique)
        return state

    def _rewrite(self, state: RagState) -> RagState:
        critique = state.get("self_rag", {})
        fallback_query = critique.get("rewrite_query") or state.get("retrieval_query") or state["sanitized_query"]
        result = self.llm.invoke_json(
            system=(
                "You are rewriting a failed insurance RAG retrieval query for a retry. Use the "
                "critique issues and original user question to create a better retrieval query. "
                "Preserve the user's facts. Do not answer the question. Return JSON only with "
                "keys: query, rationale."
            ),
            user=(
                f"Original user question:\n{state['sanitized_query']}\n\n"
                f"Previous retrieval query:\n{state.get('retrieval_query', state['sanitized_query'])}\n\n"
                f"Critique issues:\n{json.dumps(critique.get('issues', []), ensure_ascii=True)}"
            ),
            fallback={"query": fallback_query, "rationale": "fallback"},
        )
        rewrite = str(result.get("query") or fallback_query).strip() or str(fallback_query)
        state["retrieval_query"] = rewrite
        state["iteration"] = int(state.get("iteration", 0)) + 1
        self._trace(
            state,
            "rewrite",
            {
                "retrieval_query": state["retrieval_query"],
                "iteration": state["iteration"],
                "rationale": str(result.get("rationale", ""))[:300],
            },
        )
        return state

    def _finalize(self, state: RagState) -> RagState:
        if state.get("answer"):
            sanitized = self.guardrails.sanitize(state["answer"]).text
            state["answer"] = self.guardrails.clean_legacy_false_positive_placeholders(sanitized)
        if (
            state.get("use_cache", True)
            and not state.get("cache_hit")
            and state.get("answer")
            and not self._has_unsupported_citation(state)
        ):
            self.cache.save(
                query=state["query"],
                answer=state["answer"],
                confidence=float(state.get("confidence", 0.0)),
                sources=state.get("reranked_sources") or state.get("sources", []),
            )
        if state.get("answer") and state.get("reranked_sources"):
            self.memory.save_interaction(
                user_id=state.get("user_id", "default_user"),
                query=state["query"],
                answer=state["answer"],
                critique=state.get("self_rag", {}),
                sources=state.get("reranked_sources", []),
            )
        self._trace(
            state,
            "finalize",
            {
                "cache_hit": state.get("cache_hit", False),
                "confidence": state.get("confidence", 0.0),
                "iterations": state.get("iteration", 0),
            },
        )
        return state

    def _has_unsupported_citation(self, state: RagState) -> bool:
        answer = state.get("answer", "").lower()
        blocked_markers = [
            "insurance information institute",
            "source: none",
            "source: insurance",
            "according to standard insurance terminology",
        ]
        return any(marker in answer for marker in blocked_markers)

    def _route_cache(self, state: RagState) -> str:
        return "hit" if state.get("cache_hit") else "miss"

    def _route_retrieval(self, state: RagState) -> str:
        if state.get("answer") and "override system" in state["answer"]:
            return "direct"
        if state.get("intent") == "out_of_domain":
            return "out_of_domain"
        return "retrieve" if state.get("should_retrieve", True) else "direct"

    def _route_critique(self, state: RagState) -> str:
        critique = state.get("self_rag", {})
        passed = bool(critique.get("passed", False))
        confidence = float(critique.get("confidence", 0.0))
        iteration = int(state.get("iteration", 0))
        if passed and confidence >= 0.68:
            return "accept"
        if iteration >= settings.self_rag_max_loops:
            return "accept"
        if critique.get("needs_rewrite", True):
            return "retry"
        return "accept"

    def _trace(self, state: RagState, node: str, payload: dict[str, Any]) -> None:
        trace = state.setdefault("trace", [])
        trace.append({"node": node, **payload})

    def _save_trace(self, state: RagState) -> None:
        with db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO traces(request_id, query, trace_json)
                VALUES (?, ?, ?)
                """,
                (
                    state["request_id"],
                    state["query"],
                    json.dumps(state.get("trace", []), ensure_ascii=True),
                ),
            )
