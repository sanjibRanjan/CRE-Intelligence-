"""
Data Normalisation Module
=========================

Defines Pydantic v2 models that enforce a unified schema for *all*
ingested records regardless of their origin (RSS, scraping, API, CSV).

The normalisation layer sits between raw ingestion and AI processing,
guaranteeing type safety and structural consistency downstream.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════
# Supporting Models
# ═══════════════════════════════════════════════════════════════════════════


class DocumentMetadata(BaseModel):
    """Rich metadata envelope carried with every normalised document."""

    published_date: str = Field(
        default="",
        description="ISO-8601 publication / creation date string.",
    )
    source_url: str = Field(default="", description="Canonical URL of the source.")
    summary: str = Field(default="", description="LLM-generated contextual summary of the document.")
    classification: str = Field(default="", description="LLM-generated category/classification.")
    locations: list[str] = Field(
        default_factory=list,
        description="Normalised location names extracted from the text.",
    )
    entities_gpe: list[str] = Field(
        default_factory=list,
        description="Geopolitical entities (GPE) extracted via NLP.",
    )
    entities_org: list[str] = Field(
        default_factory=list,
        description="Organisation entities (ORG) extracted via NLP.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra metadata (ticker, market_cap, etc.).",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Core Normalised Document
# ═══════════════════════════════════════════════════════════════════════════


class NormalisedDocument(BaseModel):
    """Unified document schema used throughout the pipeline.

    Every piece of ingested data — whether from an RSS feed, a web scrape,
    a financial API, or a CSV row — is transformed into this model before
    any downstream AI processing takes place.
    """

    doc_id: str = Field(
        default="",
        description="Unique document identifier (auto-generated if empty).",
    )
    source_type: str = Field(
        ...,
        description="Origin of the document: rss | scraping | api | csv.",
    )
    title: str = Field(..., description="Human-readable title / headline.")
    content: str = Field(
        default="",
        description="Main text body of the document.",
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata,
        description="Structured metadata envelope.",
    )

    # ── Validators ────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _generate_doc_id(self) -> "NormalisedDocument":
        """Auto-generate a deterministic ``doc_id`` if one was not supplied.

        The ID is derived from a SHA-256 hash of ``source_type + title``
        truncated to 12 hex characters, providing collision-resistance
        while staying compact.
        """
        if not self.doc_id:
            seed = f"{self.source_type}:{self.title}"
            self.doc_id = hashlib.sha256(seed.encode()).hexdigest()[:12]
        return self

    @field_validator("source_type")
    @classmethod
    def _validate_source_type(cls, v: str) -> str:
        allowed = {"rss", "scraping", "api", "csv", "xlsx", "csv_listings"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"source_type must be one of {allowed}, got '{v}'"
            )
        return v_lower


# ═══════════════════════════════════════════════════════════════════════════
# Normalisation Functions
# ═══════════════════════════════════════════════════════════════════════════


def normalise_record(raw: dict[str, Any]) -> NormalisedDocument:
    """Convert a single raw ingestion dict into a ``NormalisedDocument``.

    The function is intentionally tolerant: missing keys fall back to
    sensible defaults rather than raising.

    Args:
        raw: Dictionary produced by one of the ingestion connectors.

    Returns:
        A validated ``NormalisedDocument`` instance.
    """
    content = raw.get("content") or raw.get("summary") or raw.get("description") or ""
    title = raw.get("title") or "Untitled"

    meta = DocumentMetadata(
        published_date=raw.get("published_date", ""),
        source_url=raw.get("link", ""),
        extra={
            k: v
            for k, v in raw.items()
            if k not in {"title", "content", "summary", "description", "link",
                         "published_date", "source_type", "doc_id"}
        },
    )

    return NormalisedDocument(
        doc_id=raw.get("doc_id", ""),
        source_type=raw.get("source_type", "csv"),
        title=title,
        content=content,
        metadata=meta,
    )


def normalise_batch(records: list[dict[str, Any]]) -> list[NormalisedDocument]:
    """Normalise a batch of raw dicts, skipping records that fail validation.

    Args:
        records: List of raw ingestion dicts.

    Returns:
        List of successfully normalised documents.
    """
    docs: list[NormalisedDocument] = []
    for rec in records:
        try:
            docs.append(normalise_record(rec))
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "Skipping invalid record (%s): %s", rec.get("title", "?"), exc
            )
    return docs
