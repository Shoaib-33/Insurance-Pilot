import json
import logging
import re
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class GroqLLM:
    def __init__(self) -> None:
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            if not settings.groq_api_key:
                return None
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(
                model=settings.groq_model,
                temperature=0,
                max_retries=2,
                api_key=settings.groq_api_key,
            )
        return self._llm

    def invoke_text(self, system: str, user: str) -> str:
        if self.llm is None:
            return self._fallback_text(user)
        response = self.llm.invoke(
            [
                ("system", system),
                ("user", user),
            ]
        )
        return str(response.content)

    def invoke_json(self, system: str, user: str, fallback: dict[str, Any]) -> dict[str, Any]:
        if self.llm is None:
            return fallback
        try:
            response = self.llm.invoke(
                [
                    ("system", system + "\nReturn only valid JSON."),
                    ("user", user),
                ],
                {"response_format": {"type": "json_object"}},
            )
            return self._parse_json(str(response.content), fallback)
        except Exception as exc:
            logger.warning("Groq JSON call failed, using fallback: %s", exc)
            return fallback

    def _parse_json(self, text: str, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return fallback
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return fallback

    def _fallback_text(self, user: str) -> str:
        return (
            "The Groq API key is not configured, so this local fallback cannot produce a full "
            "LLM answer. Add GROQ_API_KEY to .env and rerun the service."
        )
