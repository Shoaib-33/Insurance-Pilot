import re


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"developer\s+message",
    r"act\s+as\s+dan",
    r"jailbreak",
]


def detect_prompt_injection(text: str) -> list[str]:
    findings = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.I):
            findings.append(pattern)
    return findings


def strip_unsafe_retrieved_text(text: str) -> str:
    cleaned = text
    for pattern in INJECTION_PATTERNS:
        cleaned = re.sub(pattern, "[REMOVED_PROMPT_INJECTION]", cleaned, flags=re.I)
    return cleaned
