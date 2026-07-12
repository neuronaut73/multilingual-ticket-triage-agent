"""
Sprint 4 — Neighbor retrieval and weighted voting for a single eval ticket.

NeighborRetriever:
  1. Embeds the query representation_text using the embedding model.
  2. Searches LanceDB for the top-k most similar reference tickets.
  3. Converts each result row into a NeighborEvidence object.
  4. Aggregates neighbor metadata using weighted voting to produce a
     NeighborPrediction.

Only reference split rows are stored in LanceDB, so eval label leakage
is structurally impossible at this layer.
"""

from __future__ import annotations

from src.application.weighted_vote import (
    aggregate_tags,
    distance_to_similarity,
    parse_tags_json,
    weighted_vote,
)
from src.domain.models import NeighborEvidence, NeighborPrediction


class NeighborRetriever:
    """
    Retrieve top-k similar historical tickets and aggregate their metadata
    into a weighted prediction.

    Parameters
    ----------
    embedding_model:
        Any object with an `encode_queries(texts: list[str]) -> np.ndarray`
        method. In production this is EmbeddingModel. In tests it can be
        a simple fake.
    ticket_store:
        Any object with a `search(query_vector, top_k: int) -> list[dict]`
        method. In production this is LanceDBTicketStore. In tests it can
        be a simple fake.
    top_k:
        Number of neighbors to retrieve.
    """

    def __init__(self, embedding_model, ticket_store, top_k: int = 5) -> None:
        self.embedding_model = embedding_model
        self.ticket_store    = ticket_store
        self.top_k           = top_k

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        ticket_id: str,
        representation_text: str,
    ) -> list[NeighborEvidence]:
        """
        Embed representation_text and return the top-k neighbors as
        NeighborEvidence objects.

        Steps:
          1. Encode the text with the query prefix (E5 convention).
          2. Search LanceDB for top_k nearest vectors.
          3. Convert each result row to NeighborEvidence.

        Returns an empty list if the store has no data or no results.
        """
        query_vectors = self.embedding_model.encode_queries([representation_text])
        query_vec = query_vectors[0]

        raw_rows = self.ticket_store.search(query_vec, top_k=self.top_k)

        neighbors = [_row_to_evidence(row) for row in raw_rows]
        return neighbors

    def predict_from_neighbors(
        self,
        neighbors: list[NeighborEvidence],
    ) -> NeighborPrediction:
        """
        Aggregate neighbor metadata with weighted voting.

        For each label dimension (queue, priority, proxy_topic):
          - pair each neighbor's label with its similarity weight
          - call weighted_vote to get the winning label and confidence

        For tags:
          - collect (tag_list, similarity) pairs for all neighbors
          - call aggregate_tags to get the top-N tags by total weight

        Returns a NeighborPrediction with safe defaults when neighbors is
        empty or all labels are missing.
        """
        if not neighbors:
            return NeighborPrediction(
                predicted_queue=None,
                queue_confidence=0.0,
                predicted_priority=None,
                priority_confidence=0.0,
                predicted_proxy_topic=None,
                proxy_topic_confidence=0.0,
                predicted_tags=[],
                neighbors=[],
            )

        # Build (label, similarity) pairs for each dimension.
        queue_pairs    = [(nb.actual_queue,    nb.similarity) for nb in neighbors]
        priority_pairs = [(nb.actual_priority, nb.similarity) for nb in neighbors]
        topic_pairs    = [(nb.proxy_topic,     nb.similarity) for nb in neighbors]

        predicted_queue,       queue_confidence    = weighted_vote(queue_pairs)
        predicted_priority,    priority_confidence = weighted_vote(priority_pairs)
        predicted_proxy_topic, proxy_topic_confidence = weighted_vote(topic_pairs)

        # Tag aggregation: pair each neighbor's tag list with its similarity.
        tag_weight_pairs = [(nb.actual_tags, nb.similarity) for nb in neighbors]
        predicted_tags = aggregate_tags(tag_weight_pairs, top_n=5)

        return NeighborPrediction(
            predicted_queue=predicted_queue,
            queue_confidence=queue_confidence,
            predicted_priority=predicted_priority,
            priority_confidence=priority_confidence,
            predicted_proxy_topic=predicted_proxy_topic,
            proxy_topic_confidence=proxy_topic_confidence,
            predicted_tags=predicted_tags,
            neighbors=neighbors,
        )

    def retrieve_and_predict(
        self,
        ticket_id: str,
        representation_text: str,
    ) -> NeighborPrediction:
        """
        Convenience method: retrieve neighbors then aggregate into a prediction.

        This is the single entry point used by the Sprint 4 smoke run and
        the future Sprint 5 pipeline.
        """
        neighbors = self.retrieve(ticket_id, representation_text)
        return self.predict_from_neighbors(neighbors)


# ------------------------------------------------------------------
# Row conversion helper (module-level, not a method — easy to test alone)
# ------------------------------------------------------------------

def _row_to_evidence(row: dict) -> NeighborEvidence:
    """
    Convert one LanceDB result row to a NeighborEvidence object.

    LanceDB adds '_distance' to every result row. We compute similarity
    from it and keep the raw distance for tracing.

    actual_tags_json is a JSON-encoded list stored in LanceDB. We parse it
    here so the rest of the code works with plain Python lists.
    """
    distance   = float(row.get("_distance", 0.0))
    similarity = distance_to_similarity(distance)

    actual_tags = parse_tags_json(row.get("actual_tags_json", "[]"))

    return NeighborEvidence(
        ticket_id=str(row.get("ticket_id", "")),
        distance=distance,
        similarity=similarity,
        actual_queue=row.get("actual_queue") or None,
        actual_priority=row.get("actual_priority") or None,
        actual_type=row.get("actual_type") or None,
        actual_tags=actual_tags,
        proxy_topic=row.get("proxy_topic") or None,
        proxy_urgency=row.get("proxy_urgency") or None,
        proxy_next_action=row.get("proxy_next_action") or None,
        text_snippet=str(row.get("text_snippet", "")),
    )
