import json
import re
from typing import Any

from app.db.sqlite import db
from app.rag.text import new_id, normalize_text, tokenize


class ClaimMemoryService:
    """LangMem-ready memory facade with durable SQLite fallback.

    The project initializes LangMem memory tools when the package is present, while
    storing/retrieving operational memories locally so the app works without an
    external LangGraph store.
    """

    def __init__(self) -> None:
        self.langmem_available = False
        self.store = None
        self.manage_memory_tool = None
        self.search_memory_tool = None
        self._init_langmem()

    def _init_langmem(self) -> None:
        try:
            from langmem import create_manage_memory_tool, create_search_memory_tool  # noqa: F401
            from langgraph.store.memory import InMemoryStore

            self.store = InMemoryStore()
            namespace = ("claim_support_memories", "{user_id}")
            self.manage_memory_tool = create_manage_memory_tool(namespace, store=self.store)
            self.search_memory_tool = create_search_memory_tool(namespace, store=self.store)
            self.langmem_available = True
        except Exception:
            self.langmem_available = False

    def search(self, user_id: str, query: str, limit: int = 4) -> str:
        langmem_lines = self._search_langmem_store(user_id, query, limit)
        query_terms = set(tokenize(query))
        with db() as conn:
            rows = conn.execute(
                """
                SELECT kind, content, metadata_json, created_at
                FROM memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 50
                """,
                (user_id,),
            ).fetchall()

        scored = []
        for row in rows:
            content = row["content"]
            terms = set(tokenize(content))
            score = len(query_terms & terms)
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [row for _, row in scored[:limit]]
        if not selected:
            selected = rows[: min(limit, len(rows))]

        if not selected and not langmem_lines:
            return "No prior memory for this user."

        lines = []
        lines.extend(langmem_lines)
        for row in selected:
            lines.append(f"- {row['kind']}: {row['content']}")
        return "\n".join(lines)

    def save_interaction(
        self,
        user_id: str,
        query: str,
        answer: str,
        critique: dict[str, Any],
        sources: list[dict[str, Any]],
    ) -> None:
        decision = self._extract_decision(answer)
        content = (
            f"Scenario: {query} | Decision: {decision or 'unknown'} | "
            f"Confidence: {critique.get('confidence', 0.0):.2f}"
        )
        metadata = {
            "decision": decision,
            "self_rag": {
                "isrel": critique.get("isrel"),
                "issup": critique.get("issup"),
                "isuse": critique.get("isuse"),
            },
            "source_names": sorted({s.get("source_name", "unknown") for s in sources}),
        }
        self._insert_memory(user_id, "claim_interaction", content, metadata)
        self._put_langmem_store(user_id, content, metadata)

    def _search_langmem_store(self, user_id: str, query: str, limit: int) -> list[str]:
        if not self.store:
            return []
        try:
            items = self.store.search(("claim_support_memories", user_id), query=query, limit=limit)
            lines = []
            for item in items:
                value = item.value or {}
                content = value.get("content")
                if content:
                    lines.append(f"- langmem: {content}")
            return lines
        except Exception:
            return []

    def _put_langmem_store(self, user_id: str, content: str, metadata: dict[str, Any]) -> None:
        if not self.store:
            return
        try:
            self.store.put(
                ("claim_support_memories", user_id),
                key=new_id("memory"),
                value={"content": content, "metadata": metadata},
            )
        except Exception:
            return

    def _insert_memory(self, user_id: str, kind: str, content: str, metadata: dict[str, Any]) -> None:
        normalized = normalize_text(content)
        with db() as conn:
            existing = conn.execute(
                """
                SELECT memory_id FROM memories
                WHERE user_id = ? AND kind = ? AND content = ?
                """,
                (user_id, kind, content),
            ).fetchone()
            if existing:
                return
            similar = conn.execute(
                """
                SELECT memory_id FROM memories
                WHERE user_id = ? AND kind = ? AND lower(content) = ?
                """,
                (user_id, kind, normalized),
            ).fetchone()
            if similar:
                return
            conn.execute(
                """
                INSERT INTO memories(memory_id, user_id, kind, content, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (new_id("memory"), user_id, kind, content, json.dumps(metadata, ensure_ascii=True)),
            )

    def _extract_decision(self, answer: str) -> str | None:
        match = re.search(r"Decision:\s*(.+)", answer, flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip().rstrip(".")
