import re
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    text: str
    blocked: bool
    findings: list[str]


def build_langchain_pii_middlewares() -> list[object]:
    """Create LangChain PII middleware objects for agent-compatible deployments."""
    try:
        from langchain.agents.middleware import PIIMiddleware

        return [
            PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_output=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True, apply_to_output=True),
            PIIMiddleware("ip", strategy="redact", apply_to_input=True, apply_to_output=True),
            PIIMiddleware(
                "policy_number",
                detector=r"\b(?:POL[-\s]?(?=[A-Z0-9-]*\d)[A-Z0-9]{5,}|Policy[-\s]*(?:ID|No\.?|Number)[-:\s]*[A-Z0-9]*\d[A-Z0-9-]{4,})\b",
                strategy="hash",
                apply_to_input=True,
                apply_to_output=True,
            ),
            PIIMiddleware(
                "claim_id",
                detector=r"\b(?:CLM[-\s]?(?=[A-Z0-9-]*\d)[A-Z0-9]{5,}|Claim[-\s]?(?:ID|No\.?|Number)?[-\s]*[A-Z0-9]*\d[A-Z0-9-]{4,})\b",
                strategy="hash",
                apply_to_input=True,
                apply_to_output=True,
            ),
        ]
    except Exception:
        return []


class PIIGuardrails:
    def __init__(self) -> None:
        self.langchain_middlewares = build_langchain_pii_middlewares()
        self.patterns: list[tuple[str, re.Pattern[str], str]] = [
            ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[REDACTED_EMAIL]"),
            ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[MASKED_CARD]"),
            ("phone", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
            (
                "policy_number",
                re.compile(r"\b(?:POL[-\s]?(?=[A-Z0-9-]*\d)[A-Z0-9]{5,}|Policy[-\s]*(?:ID|No\.?|Number)[-:\s]*[A-Z0-9]*\d[A-Z0-9-]{4,})\b", re.I),
                "[HASHED_POLICY_NUMBER]",
            ),
            (
                "claim_id",
                re.compile(r"\b(?:CLM[-\s]?(?=[A-Z0-9-]*\d)[A-Z0-9]{5,}|Claim[-\s]?(?:ID|No\.?|Number)?[-\s]*[A-Z0-9]*\d[A-Z0-9-]{4,})\b", re.I),
                "[HASHED_CLAIM_ID]",
            ),
        ]

    def sanitize(self, text: str) -> GuardrailResult:
        findings: list[str] = []
        sanitized = text
        for name, pattern, replacement in self.patterns:
            if pattern.search(sanitized):
                findings.append(name)
                sanitized = pattern.sub(replacement, sanitized)
        return GuardrailResult(text=sanitized, blocked=False, findings=findings)

    def clean_legacy_false_positive_placeholders(self, text: str) -> str:
        """Repair old cached answers where normal policy words were over-masked."""
        cleaned = text
        replacements = [
            (r"standard insurance \[HASHED_POLICY_NUMBER\]", "standard insurance policy"),
            (r"insurance \[HASHED_POLICY_NUMBER\]", "insurance policy"),
            (r"\b[Tt]he \[HASHED_POLICY_NUMBER\]'s", "the policyholder's"),
            (r"\b[Tt]he \[HASHED_POLICY_NUMBER\]", "the policyholder"),
            (r"\[HASHED_POLICY_NUMBER\]'s specific policy documents", "the policyholder's specific policy documents"),
            (r"\[HASHED_POLICY_NUMBER\] documents", "policy documents"),
            (r"\[HASHED_POLICY_NUMBER\] or endorsements", "policy documents or endorsements"),
        ]
        for pattern, replacement in replacements:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned
