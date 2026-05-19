import hashlib
import re
import uuid


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_id(prefix: str) -> str:
    return str(uuid.uuid4())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 140) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def simhash(text: str, bits: int = 64) -> int:
    tokens = tokenize(normalize_text(text))
    vector = [0] * bits
    for token in tokens:
        digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            vector[i] += 1 if digest & (1 << i) else -1
    value = 0
    for i, weight in enumerate(vector):
        if weight > 0:
            value |= 1 << i
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()
