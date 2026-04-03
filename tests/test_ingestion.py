"""
Tests — Data Ingestion
======================

Verifies the ingestion connectors return correctly shaped data and
handle edge cases gracefully.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.ingestion import (
    fetch_fmp_profiles,
    ingest_rss,
    load_cities_csv,
    load_cre_lending_data,
    load_homes_csv,
    load_zillow_csv,
    scrape_jll_articles,
    scrape_altus_articles,
)
from src.config import (
    CITIES_CSV_PATH,
    CRE_LENDING_XLSX_PATH,
    HOMES_CSV_PATH,
    ZILLOW_CSV_PATH,
)

# ── Mock Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_requests_get():
    with patch("src.ingestion.requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_feedparser_parse():
    with patch("src.ingestion.feedparser.parse") as mock_parse:
        yield mock_parse


# ── RSS Tests ─────────────────────────────────────────────────────────────

class TestRSSIngestion:
    """Tests for the RSS feed ingestion connector."""

    def test_ingest_rss_empty_on_error(self, mock_feedparser_parse) -> None:
        """ingest_rss should return empty list if bozo error."""
        mock_feed = MagicMock()
        mock_feed.bozo = 1
        mock_feed.entries = []
        mock_feedparser_parse.return_value = mock_feed
        results = ingest_rss()
        assert results == []

    def test_ingest_rss_success(self, mock_feedparser_parse) -> None:
        """ingest_rss should return parsed entries."""
        mock_feed = MagicMock()
        mock_feed.bozo = 0
        mock_feed.entries = [{"title": "Test", "link": "http://x", "published": "2026-01-01", "summary": "Sum"}]
        mock_feedparser_parse.return_value = mock_feed
        results = ingest_rss()
        assert len(results) == 1
        assert results[0]["source_type"] == "rss"


# ── JLL Scraping Tests ────────────────────────────────────────────────────

class TestJLLScraping:
    """Tests for the JLL article scraping connector."""

    def test_jll_empty_on_error(self, mock_requests_get) -> None:
        """scrape_jll_articles should return empty list on failure."""
        mock_requests_get.side_effect = Exception("Network Error")
        results = scrape_jll_articles()
        assert results == []


# ── Altus Group Scraping Tests ────────────────────────────────────────────

class TestAltusScraping:
    """Tests for the Altus Group article scraping connector."""

    def test_altus_empty_on_error(self, mock_requests_get) -> None:
        """scrape_altus_articles should return empty list on failure."""
        mock_requests_get.side_effect = Exception("Network Error")
        results = scrape_altus_articles()
        assert results == []


# ── FMP API Tests ─────────────────────────────────────────────────────────

class TestFMPAPI:
    """Tests for the Financial Modeling Prep API connector."""

    def test_fmp_empty_on_error(self, mock_requests_get) -> None:
        """fetch_fmp_profiles should be fully robust against errors."""
        mock_requests_get.side_effect = Exception("Network Error")
        profiles = fetch_fmp_profiles(tickers=["CBRE"])
        assert len(profiles) == 0

    def test_fetch_fmp_success(self, mock_requests_get) -> None:
        """fetch_fmp_profiles should return a profile."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"companyName": "CBRE Group", "sector": "Real Estate"}]
        mock_requests_get.return_value = mock_resp
        profiles = fetch_fmp_profiles(tickers=["CBRE"])
        assert len(profiles) == 1
        assert "CBRE Group" in profiles[0]["title"]


# ── CSV Tests ─────────────────────────────────────────────────────────────

class TestCSVIngestion:
    """Tests for the cities, homes, and zillow CSV loaders."""

    def test_load_cities_csv_from_real_data(self) -> None:
        """Load the actual cities.csv and verify structure."""
        cities = load_cities_csv(CITIES_CSV_PATH)
        assert isinstance(cities, list)
        assert len(cities) > 50  # real file has ~129 rows
        first_city = cities[0]
        assert "city" in first_city or "City" in first_city

    def test_cities_csv_has_state_info(self) -> None:
        """Each city row should have state info."""
        cities = load_cities_csv(CITIES_CSV_PATH)
        for city in cities[:5]:
            assert "state" in city or "State" in city

    def test_load_homes_csv(self) -> None:
        """Load homes.csv and verify records are returned."""
        homes = load_homes_csv(HOMES_CSV_PATH)
        assert isinstance(homes, list)
        assert len(homes) > 0
        assert homes[0]["source_type"] == "csv_listings"
        assert "sell_price" in homes[0]

    def test_load_zillow_csv(self) -> None:
        """Load zillow.csv and verify records are returned."""
        zillow = load_zillow_csv(ZILLOW_CSV_PATH)
        assert isinstance(zillow, list)
        assert len(zillow) > 0
        assert zillow[0]["source_type"] == "csv_listings"
        assert "list_price" in zillow[0]

    def test_homes_csv_missing_file(self, tmp_path) -> None:
        """Loading a non-existent homes file should return empty list."""
        result = load_homes_csv(tmp_path / "nonexistent.csv")
        assert result == []


# ── XLSX Tests ────────────────────────────────────────────────────────────

class TestCRELendingData:
    """Tests for the CRE Lending XLSX ingestion."""

    def test_load_cre_lending_returns_data(self) -> None:
        """load_cre_lending_data should parse deals from both sheets."""
        deals = load_cre_lending_data(CRE_LENDING_XLSX_PATH)
        assert isinstance(deals, list)
        assert len(deals) > 10

    def test_lending_deal_has_required_fields(self) -> None:
        """Each lending deal should contain key fields."""
        deals = load_cre_lending_data(CRE_LENDING_XLSX_PATH)
        required_fields = {"title", "content", "source_type", "lender", "borrower", "loan_size_m"}
        for deal in deals[:5]:
            assert required_fields.issubset(deal.keys()), (
                f"Missing fields: {required_fields - deal.keys()}"
            )

    def test_lending_source_type_is_xlsx(self) -> None:
        """Every lending deal should have source_type == 'xlsx'."""
        deals = load_cre_lending_data(CRE_LENDING_XLSX_PATH)
        for deal in deals:
            assert deal["source_type"] == "xlsx"

    def test_lending_deals_have_region(self) -> None:
        """Deals should be tagged with UK or Continental Europe region."""
        deals = load_cre_lending_data(CRE_LENDING_XLSX_PATH)
        regions = {d["region"] for d in deals}
        assert "UK" in regions or "Continental Europe" in regions

    def test_lending_missing_file(self, tmp_path) -> None:
        """Loading a non-existent XLSX should return empty list."""
        result = load_cre_lending_data(tmp_path / "nonexistent.xlsx")
        assert result == []
