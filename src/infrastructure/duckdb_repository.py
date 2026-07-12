import hashlib
import json
from datetime import datetime, timezone

import duckdb

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS historical_tickets (
    ticket_id              TEXT PRIMARY KEY,
    source_row_index       INTEGER,
    split_name             TEXT NOT NULL,       -- 'reference' or 'eval'
    subject                TEXT NOT NULL,
    body                   TEXT NOT NULL,
    raw_text               TEXT NOT NULL,
    cleaned_text           TEXT NOT NULL,
    representation_text    TEXT NOT NULL,
    text_snippet           TEXT,
    actual_queue           TEXT,
    actual_priority        TEXT,
    actual_type            TEXT,
    actual_tags_json       TEXT,               -- JSON array of tag strings
    language               TEXT,
    proxy_topic            TEXT,               -- derived assignment topic
    proxy_urgency          TEXT,               -- derived urgency
    proxy_next_action      TEXT,               -- derived default next action
    proxy_topic_source     TEXT,               -- strong_tag_signal | queue_mapping | fallback_other
    source_row_json        TEXT,               -- original CSV row as JSON; answer excluded
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

INSERT_SQL = """
INSERT OR REPLACE INTO historical_tickets (
    ticket_id, source_row_index, split_name,
    subject, body, raw_text, cleaned_text, representation_text, text_snippet,
    actual_queue, actual_priority, actual_type, actual_tags_json, language,
    proxy_topic, proxy_urgency, proxy_next_action, proxy_topic_source,
    source_row_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _deterministic_ticket_id(subject: str, body: str, row_index: int) -> str:
    """
    Stable ID derived from content + original row index.
    Using a hash means the same CSV always produces the same IDs.
    """
    raw = f"{row_index}||{subject}||{body}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _assign_split(ticket_id: str, eval_fraction: float, random_seed: int) -> str:
    """
    Deterministic split: hash the id + seed, map to [0, 1), compare to fraction.
    No randomness at call time — same inputs always yield the same split.
    """
    raw = f"{random_seed}||{ticket_id}"
    digest = hashlib.sha1(raw.encode()).digest()
    # First 4 bytes as unsigned int, normalised to [0, 1)
    value = int.from_bytes(digest[:4], "big") / 2**32
    return "eval" if value < eval_fraction else "reference"


class DuckDBRepository:
    """
    Thin wrapper around a DuckDB connection for historical_tickets storage.
    Holds only the connection as state.
    """

    def __init__(self, db_path: str = "data/tickets.duckdb", recreate: bool = False):
        self.conn = duckdb.connect(db_path)
        if recreate:
            self.conn.execute("DROP TABLE IF EXISTS historical_tickets")
        self.conn.execute(CREATE_TABLE_SQL)

    def close(self):
        self.conn.close()

    def insert_tickets(
        self,
        rows: list[dict],
        eval_fraction: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        """
        Insert pre-processed ticket dicts into historical_tickets.

        Each dict must have:
            subject, body,
            raw_text, cleaned_text, representation_text, text_snippet,
            actual_queue, actual_priority, actual_type,
            actual_tags_json, language,
            proxy_topic, proxy_urgency, proxy_next_action, proxy_topic_source,
            source_row_json,
            _row_index  (used only for id generation, not stored directly)

        source_row_json must not contain the answer column; callers are
        responsible for excluding it before building this dict.
        """
        records = []
        for row in rows:
            ticket_id = _deterministic_ticket_id(
                row["subject"], row["body"], row["_row_index"]
            )
            split_name = _assign_split(ticket_id, eval_fraction, random_seed)
            records.append((
                ticket_id,
                row["_row_index"],
                split_name,
                row["subject"],
                row["body"],
                row["raw_text"],
                row["cleaned_text"],
                row["representation_text"],
                row["text_snippet"],
                row["actual_queue"],
                row["actual_priority"],
                row["actual_type"],
                row["actual_tags_json"],
                row["language"],
                row["proxy_topic"],
                row["proxy_urgency"],
                row["proxy_next_action"],
                row["proxy_topic_source"],
                row["source_row_json"],
            ))
        self.conn.executemany(INSERT_SQL, records)

    def count_by_split(self) -> dict[str, int]:
        """Return {split_name: row_count} for all splits in the table."""
        result = self.conn.execute(
            "SELECT split_name, COUNT(*) FROM historical_tickets GROUP BY split_name"
        ).fetchall()
        return {split: count for split, count in result}

    def fetch_split_tickets(self, split: str) -> list[dict]:
        """
        Return all tickets for the given split with every column needed for
        batch processing and post-prediction evaluation.

        Rows are ordered by ticket_id so the natural strategy is deterministic.
        No LIMIT is applied — callers select the desired subset after fetching
        using _sample_tickets() in main.py.

        Data leakage note:
          proxy_* and actual_* columns are included as evaluation metadata.
          They must not be placed into TicketInput — that responsibility lies
          with BatchRunner, which builds TicketInput from text fields only.
        """
        result = self.conn.execute(
            """
            SELECT
                ticket_id, subject, body, raw_text, cleaned_text,
                representation_text, text_snippet,
                actual_queue, actual_priority, actual_type, actual_tags_json,
                proxy_topic, proxy_urgency, proxy_next_action, proxy_topic_source
            FROM historical_tickets
            WHERE split_name = ?
            ORDER BY ticket_id
            """,
            [split],
        ).fetchall()
        columns = [
            "ticket_id", "subject", "body", "raw_text", "cleaned_text",
            "representation_text", "text_snippet",
            "actual_queue", "actual_priority", "actual_type", "actual_tags_json",
            "proxy_topic", "proxy_urgency", "proxy_next_action", "proxy_topic_source",
        ]
        return [dict(zip(columns, row)) for row in result]

    def count_by_proxy_topic(self) -> dict[str, int]:
        """Return {proxy_topic: row_count} sorted by count descending."""
        result = self.conn.execute(
            "SELECT proxy_topic, COUNT(*) AS n FROM historical_tickets"
            " GROUP BY proxy_topic ORDER BY n DESC"
        ).fetchall()
        return {topic: count for topic, count in result}

    def sample_rows(self, n: int = 5) -> list[dict]:
        """
        Return up to n rows as dicts with the key fields for quick inspection.

        Uses USING SAMPLE to pick rows without a full table scan.
        """
        result = self.conn.execute(
            f"""
            SELECT
                ticket_id,
                split_name,
                subject,
                actual_queue,
                actual_priority,
                proxy_topic,
                proxy_urgency,
                proxy_next_action,
                proxy_topic_source
            FROM historical_tickets
            USING SAMPLE {n}
            """
        ).fetchall()
        columns = [
            "ticket_id", "split_name", "subject",
            "actual_queue", "actual_priority",
            "proxy_topic", "proxy_urgency", "proxy_next_action", "proxy_topic_source",
        ]
        return [dict(zip(columns, row)) for row in result]
