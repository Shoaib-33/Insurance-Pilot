import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.core.config import settings


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    normalized_hash TEXT NOT NULL,
    simhash TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    embedded_at TEXT,
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS answer_cache (
    cache_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    normalized_query TEXT NOT NULL,
    answer TEXT NOT NULL,
    confidence REAL NOT NULL,
    sources_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS traces (
    request_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    original_answer TEXT NOT NULL,
    approved_answer TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    preferred_contact TEXT NOT NULL,
    risk_notes TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    policy_type TEXT NOT NULL,
    active INTEGER NOT NULL,
    coverages_json TEXT NOT NULL,
    exclusions_json TEXT NOT NULL,
    deductible REAL NOT NULL,
    policy_limit REAL NOT NULL,
    endorsements_json TEXT NOT NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    status TEXT NOT NULL,
    date_of_loss TEXT NOT NULL,
    missing_documents_json TEXT NOT NULL,
    notes TEXT NOT NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY(policy_id) REFERENCES policies(policy_id)
);

CREATE TABLE IF NOT EXISTS ticket_queues (
    queue_name TEXT PRIMARY KEY,
    open_tickets INTEGER NOT NULL,
    estimated_review_time TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash, normalized_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(text_hash);
CREATE INDEX IF NOT EXISTS idx_memories_user_kind ON memories(user_id, kind, created_at);
"""


SEED_SQL = """
INSERT OR IGNORE INTO customers(customer_id, name, preferred_contact, risk_notes)
VALUES
('CUS-1001', 'Maya Rahman', 'email', 'Prior water claim required mitigation documentation.'),
('CUS-1002', 'Omar Khan', 'phone', 'No special risk notes.'),
('CUS-1003', 'Nadia Islam', 'email', 'Previously submitted theft claim with missing receipts.');

INSERT OR IGNORE INTO policies(policy_id, customer_id, policy_type, active, coverages_json, exclusions_json, deductible, policy_limit, endorsements_json)
VALUES
('POL-3001', 'CUS-1001', 'property', 1, '["sudden accidental water discharge", "fire and smoke", "wind and hail", "theft of personal property"]', '["gradual leakage", "mold from long-term seepage", "flood without endorsement", "wear and tear"]', 1000, 25000, '["limited sewer backup"]'),
('POL-3002', 'CUS-1002', 'auto', 1, '["collision", "comprehensive theft", "liability"]', '["intentional damage", "unlisted commercial use"]', 500, 40000, '[]'),
('POL-3003', 'CUS-1003', 'property', 1, '["fire and smoke", "theft of personal property", "wind and hail"]', '["flood", "gradual leakage", "high-value unscheduled jewelry above sublimit"]', 1500, 50000, '["scheduled jewelry required above sublimit"]');

INSERT OR IGNORE INTO claims(claim_id, customer_id, policy_id, claim_type, status, date_of_loss, missing_documents_json, notes)
VALUES
('CLM-1007', 'CUS-1001', 'POL-3001', 'water_damage', 'documents_pending', '2026-05-10', '["mitigation invoice", "repair estimate"]', 'Kitchen burst pipe; photos and plumber report received.'),
('CLM-2011', 'CUS-1003', 'POL-3003', 'theft', 'human_review', '2026-04-28', '["receipts", "serial numbers"]', 'Laptop and camera stolen from vehicle; police report received.'),
('CLM-3020', 'CUS-1001', 'POL-3001', 'storm', 'new', '2026-05-12', '["contractor estimate", "weather event confirmation"]', 'Roof hail damage reported.');

INSERT OR IGNORE INTO ticket_queues(queue_name, open_tickets, estimated_review_time)
VALUES
('property_claims', 14, '1 business day'),
('auto_claims', 8, 'same day'),
('special_investigation', 6, '2 business days'),
('liability_claims', 11, '1-2 business days');
"""


def connect() -> sqlite3.Connection:
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript(SCHEMA)
        conn.executescript(SEED_SQL)
