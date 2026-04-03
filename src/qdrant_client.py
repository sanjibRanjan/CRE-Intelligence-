"""
Qdrant Vector Database Client
==============================

Handles all interactions with the Qdrant vector database:
- Collection initialisation (create-if-not-exists)
- Upserting chunk records (vectors + payloads)
- Semantic search with optional payload filtering
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import (
    EMBEDDING_DIMENSION,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Client Factory
# ═══════════════════════════════════════════════════════════════════════════

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant client instance.

    Falls back to an in-memory client when the Qdrant server is
    unreachable, allowing local development without Docker.

    Returns:
        A ``QdrantClient`` instance.
    """
    global _client  # noqa: PLW0603
    if _client is None:
        try:
            _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
            # Quick health check
            _client.get_collections()
            logger.info("Connected to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)
        except Exception:
            logger.warning(
                "Qdrant server unreachable at %s:%s — using in-memory mode",
                QDRANT_HOST,
                QDRANT_PORT,
            )
            _client = QdrantClient(location=":memory:")
    return _client


# ═══════════════════════════════════════════════════════════════════════════
# Collection Management
# ═══════════════════════════════════════════════════════════════════════════


def init_collection(
    collection_name: str = QDRANT_COLLECTION,
    vector_size: int = EMBEDDING_DIMENSION,
    recreate: bool = False,
) -> None:
    """Initialise a Qdrant collection, creating it if it does not exist.

    Args:
        collection_name: Name of the collection.
        vector_size: Dimensionality of the embedding vectors.
        recreate: If ``True``, drop and recreate the collection.
    """
    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]

    if recreate and collection_name in existing:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'", collection_name)
        existing.remove(collection_name)

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection '%s' (dim=%d, cosine)",
            collection_name,
            vector_size,
        )
    else:
        logger.info("Collection '%s' already exists — reusing", collection_name)


# ═══════════════════════════════════════════════════════════════════════════
# Upsert
# ═══════════════════════════════════════════════════════════════════════════


def upsert_records(
    records: list[dict[str, Any]],
    collection_name: str = QDRANT_COLLECTION,
    batch_size: int = 100,
) -> int:
    """Upsert chunk records into Qdrant.

    Each record must contain an ``embedding`` key (list[float]) and
    arbitrary payload fields.

    Args:
        records: Chunk records produced by the AI processor.
        collection_name: Target Qdrant collection.
        batch_size: Number of points per upsert call.

    Returns:
        Total number of points upserted.
    """
    client = get_qdrant_client()
    points: list[PointStruct] = []

    for rec in records:
        embedding = rec.pop("embedding", None)
        if embedding is None:
            logger.warning("Record missing embedding — skipping: %s", rec.get("chunk_id"))
            continue

        # Build a clean JSON-serialisable payload
        payload: dict[str, Any] = {}
        for key, value in rec.items():
            payload[key] = value

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, rec.get("chunk_id", str(uuid.uuid4()))))
        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    # Batch upsert
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        total += len(batch)

    logger.info("Upserted %d points into '%s'", total, collection_name)
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval / Search
# ═══════════════════════════════════════════════════════════════════════════


def search(
    query_vector: list[float],
    collection_name: str = QDRANT_COLLECTION,
    top_k: int = 5,
    source_type: str | None = None,
    city: str | None = None,
) -> list[dict[str, Any]]:
    """Perform semantic search in Qdrant with optional payload filters.

    Args:
        query_vector: The query embedding vector.
        collection_name: Qdrant collection to search.
        top_k: Number of results to return.
        source_type: Optional filter on ``source_type`` payload field.
        city: Optional filter — only return results whose ``locations``
              list contains this city name.

    Returns:
        A list of result dicts, each containing ``score``, ``payload``,
        and ``id``.
    """
    client = get_qdrant_client()

    conditions: list[FieldCondition] = []
    if source_type:
        conditions.append(
            FieldCondition(key="source_type", match=MatchValue(value=source_type))
        )
    if city:
        conditions.append(
            FieldCondition(key="locations", match=MatchValue(value=city))
        )

    query_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    )

    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in results.points
    ]


def get_all_payloads(
    collection_name: str = QDRANT_COLLECTION,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Retrieve all payloads from a collection (for dashboard analytics).

    Args:
        collection_name: Qdrant collection name.
        limit: Maximum number of points to scroll.

    Returns:
        A list of payload dicts.
    """
    client = get_qdrant_client()
    try:
        results = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points = results[0] if results else []
        return [p.payload for p in points if p.payload]
    except Exception as exc:
        logger.error("Failed to scroll payloads: %s", exc)
        return []
