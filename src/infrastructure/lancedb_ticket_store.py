"""
Sprint 3 — LanceDB vector store for historical reference tickets.

Stores one vector per reference ticket alongside metadata needed for
retrieval evidence and evaluation. Only reference split rows must be inserted.
Eval rows must never be indexed here.
"""

import lancedb


class LanceDBTicketStore:
    """
    Thin wrapper around a LanceDB table for ticket vector storage and search.

    The table schema is inferred from the first batch of rows passed to
    recreate_table. Each row must contain:
        vector            – list[float] of length == embedding_dimension
        ticket_id         – str
        split_name        – str  (always 'reference' when indexing)
        representation_text – str
        text_snippet      – str
        actual_queue      – str
        actual_priority   – str
        actual_type       – str
        actual_tags_json  – str  (JSON array)
        proxy_topic       – str
        proxy_urgency     – str
        proxy_next_action – str
        proxy_topic_source – str
    """

    def __init__(self, path: str, table_name: str) -> None:
        self._db = lancedb.connect(path)
        self._table_name = table_name
        self._table = None  # opened / set by recreate_table or _open_existing

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def recreate_table(self, rows: list[dict]) -> None:
        """
        Drop the existing table (if any) and create a fresh one from rows.

        Using mode='overwrite' means the same call is idempotent for reruns.
        Each dict in rows must have a 'vector' key plus the metadata fields
        listed in the class docstring.
        """
        if not rows:
            raise ValueError("Cannot create an empty LanceDB table: rows list is empty.")
        self._table = self._db.create_table(
            self._table_name,
            data=rows,
            mode="overwrite",
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query_vector, top_k: int = 5) -> list[dict]:
        """
        Return the top_k most similar rows to query_vector.

        Results include all stored metadata columns plus '_distance'
        (lower is more similar for L2; for normalised cosine vectors this
        equals 2*(1 - cosine_similarity)).

        Returns an empty list if the table has not been created yet.
        """
        table = self._get_table()
        if table is None:
            return []
        return table.search(query_vector).limit(top_k).to_list()

    def count(self) -> int:
        """Return the number of rows in the table, or 0 if the table does not exist."""
        table = self._get_table()
        if table is None:
            return 0
        return table.count_rows()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_table(self):
        """Return the cached table handle, or try to open it if not yet loaded."""
        if self._table is not None:
            return self._table
        if self._table_name in self._db.table_names():
            self._table = self._db.open_table(self._table_name)
            return self._table
        return None
