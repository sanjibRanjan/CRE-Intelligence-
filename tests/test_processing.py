"""
Tests — Data Processing & Normalisation
========================================

Verifies Pydantic normalisation, entity extraction, text chunking,
and the end-to-end processing pipeline.
"""

from __future__ import annotations

import pytest

from src.ai_processor import chunk_text, normalise_locations
from src.normalization import NormalisedDocument, normalise_batch, normalise_record


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def raw_rss_record() -> dict:
    """Fixture providing a raw RSS-style ingestion dict."""
    return {
        "title": "London Office Market Shows Recovery",
        "link": "https://example.com/london-recovery",
        "published_date": "2026-03-28T10:00:00+00:00",
        "summary": (
            "The London office market is recovering. CBRE reports strong "
            "demand in Canary Wharf and the City. Manchester and Birmingham "
            "also show positive trends."
        ),
        "source_type": "rss",
    }


@pytest.fixture
def raw_api_record() -> dict:
    """Fixture providing a raw API-style ingestion dict."""
    return {
        "title": "CBRE Group Inc — Company Profile",
        "content": (
            "CBRE Group is the world's largest commercial real estate firm. "
            "HQ: Dallas, United States. Operations in London, Paris, Tokyo."
        ),
        "source_type": "api",
        "ticker": "CBRE",
        "market_cap": 38_500_000_000,
    }


@pytest.fixture
def cities_lookup() -> list[dict[str, str]]:
    """Fixture providing a small cities lookup table."""
    return [
        {"city": "London", "country": "United Kingdom", "region": "Europe"},
        {"city": "Manchester", "country": "United Kingdom", "region": "Europe"},
        {"city": "Birmingham", "country": "United Kingdom", "region": "Europe"},
        {"city": "Paris", "country": "France", "region": "Europe"},
        {"city": "Tokyo", "country": "Japan", "region": "Asia-Pacific"},
        {"city": "Dallas", "country": "United States", "region": "North America"},
        {"city": "New York", "country": "United States", "region": "North America"},
    ]


# ── Normalisation Tests ──────────────────────────────────────────────────


class TestNormalisation:
    """Tests for the Pydantic normalisation layer."""

    def test_normalise_record_rss(self, raw_rss_record: dict) -> None:
        """Normalising an RSS record should produce a valid NormalisedDocument."""
        doc = normalise_record(raw_rss_record)
        assert isinstance(doc, NormalisedDocument)
        assert doc.source_type == "rss"
        assert doc.title == "London Office Market Shows Recovery"
        assert len(doc.doc_id) > 0

    def test_normalise_record_api(self, raw_api_record: dict) -> None:
        """Normalising an API record should preserve extra metadata."""
        doc = normalise_record(raw_api_record)
        assert doc.source_type == "api"
        assert "CBRE" in doc.title
        assert doc.metadata.extra.get("ticker") == "CBRE"

    def test_normalise_record_generates_doc_id(self, raw_rss_record: dict) -> None:
        """A doc_id should be auto-generated if not provided."""
        doc = normalise_record(raw_rss_record)
        assert doc.doc_id
        assert len(doc.doc_id) == 12

    def test_normalise_record_content_from_summary(self, raw_rss_record: dict) -> None:
        """When 'content' is absent, 'summary' should be used."""
        doc = normalise_record(raw_rss_record)
        assert "London office market" in doc.content

    def test_normalise_batch_skips_invalid(self) -> None:
        """normalise_batch should skip records that fail validation."""
        records = [
            {"title": "Valid", "source_type": "rss", "summary": "ok"},
            {"title": "Invalid Source", "source_type": "INVALID_TYPE"},
        ]
        docs = normalise_batch(records)
        assert len(docs) == 1
        assert docs[0].title == "Valid"

    def test_normalised_document_invalid_source_type_raises(self) -> None:
        """Creating a NormalisedDocument with invalid source_type should raise."""
        with pytest.raises(Exception):
            NormalisedDocument(
                source_type="unknown",
                title="Test",
                content="Test content",
            )

    def test_normalise_xlsx_record(self) -> None:
        """A CRE lending deal with source_type 'xlsx' should normalise."""
        raw = {
            "title": "OakNorth → Ankor: £17.5m",
            "content": "CRE lending deal. Lender: OakNorth.",
            "source_type": "xlsx",
            "lender": "OakNorth",
            "borrower": "Ankor Property Group",
            "loan_size_m": 17.5,
        }
        doc = normalise_record(raw)
        assert doc.source_type == "xlsx"
        assert "OakNorth" in doc.title

    def test_normalise_csv_listings_record(self) -> None:
        """A property listing with source_type 'csv_listings' should normalise."""
        raw = {
            "title": "Residential Property #1",
            "content": "3 bedrooms, 2 bathrooms, list price $180K.",
            "source_type": "csv_listings",
            "dataset": "homes",
        }
        doc = normalise_record(raw)
        assert doc.source_type == "csv_listings"


# ── Location Normalisation Tests ──────────────────────────────────────────


class TestLocationNormalisation:
    """Tests for cross-referencing GPEs with cities.csv."""

    def test_normalise_known_cities(self, cities_lookup: list[dict[str, str]]) -> None:
        """Known cities should be normalised to their canonical form."""
        gpes = ["london", "PARIS", "Tokyo"]
        result = normalise_locations(gpes, cities_lookup)
        assert "London" in result
        assert "Paris" in result
        assert "Tokyo" in result

    def test_normalise_unknown_cities_filtered(self, cities_lookup: list[dict[str, str]]) -> None:
        """Unknown cities should be filtered out."""
        gpes = ["Atlantis", "Narnia"]
        result = normalise_locations(gpes, cities_lookup)
        assert len(result) == 0

    def test_normalise_deduplicates(self, cities_lookup: list[dict[str, str]]) -> None:
        """Duplicate GPE entries should be deduplicated."""
        gpes = ["London", "london", "LONDON"]
        result = normalise_locations(gpes, cities_lookup)
        assert result.count("London") == 1

    def test_normalise_with_real_csv_format(self) -> None:
        """Normalise locations using the real cities.csv format (City key)."""
        # Simulate the actual CSV format with 'City' and 'State' keys
        real_format_lookup = [
            {"City": "Washington", "State": "DC", "city": "Washington", "state": "DC"},
            {"City": "Chicago", "State": "IL", "city": "Chicago", "state": "IL"},
            {"City": "Dallas", "State": "TX", "city": "Dallas", "state": "TX"},
        ]
        gpes = ["Washington", "chicago", "Unknown City"]
        result = normalise_locations(gpes, real_format_lookup)
        assert "Washington" in result
        assert "Chicago" in result
        assert len(result) == 2


# ── Text Chunking Tests ──────────────────────────────────────────────────


class TestTextChunking:
    """Tests for the text chunking function."""

    def test_chunk_short_text(self) -> None:
        """Text shorter than chunk_size should produce a single chunk."""
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=300)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text(self) -> None:
        """Long text should produce multiple chunks."""
        words = ["word"] * 900
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=300)
        assert len(chunks) >= 3

    def test_chunk_empty_text(self) -> None:
        """Empty text should return an empty list."""
        chunks = chunk_text("")
        assert chunks == []

    def test_chunk_overlap(self) -> None:
        """Consecutive chunks should share some overlapping words."""
        words = [f"w{i}" for i in range(600)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=300)
        assert len(chunks) >= 2
        # The last words of chunk 0 should overlap with the first words of chunk 1
        c0_words = chunks[0].split()
        c1_words = chunks[1].split()
        overlap = set(c0_words[-30:]) & set(c1_words[:30:])
        assert len(overlap) > 0
