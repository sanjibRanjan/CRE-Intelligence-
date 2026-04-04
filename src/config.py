"""
Configuration module for the AI Data Engineer pipeline.

Centralises all environment variables, API keys, model paths, and
application-level constants so that every other module imports from
a single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: load .env file if present (local dev convenience)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CITIES_CSV_PATH: Path = DATA_DIR / "cities.csv"
HOMES_CSV_PATH: Path = DATA_DIR / "homes.csv"
ZILLOW_CSV_PATH: Path = DATA_DIR / "zillow.csv"
CRE_LENDING_XLSX_PATH: Path = DATA_DIR / "Real-Estate-Capital-Europe-Sample-CRE-Lending-Data.xlsx"

# ---------------------------------------------------------------------------
# RSS Feed Configuration
# ---------------------------------------------------------------------------
RSS_FEED_URL: str = os.getenv(
    "RSS_FEED_URL",
    "https://www.ft.com/property?format=rss",
)

# ---------------------------------------------------------------------------
# Scraping Configuration
# ---------------------------------------------------------------------------
JLL_BASE_URL: str = os.getenv(
    "JLL_BASE_URL",
    "https://www.jll.co.uk/en/trends-and-insights",
)
ALTUS_BASE_URL: str = os.getenv(
    "ALTUS_BASE_URL",
    "https://www.altusgroup.com/insights/",
)
SCRAPE_MAX_ARTICLES: int = int(os.getenv("SCRAPE_MAX_ARTICLES", "15"))
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Financial Modeling Prep (FMP) API
# ---------------------------------------------------------------------------
FMP_API_KEY: str = os.getenv("FMP_API_KEY", "demo")
FMP_BASE_URL: str = "https://financialmodelingprep.com/stable"
FMP_TICKERS: list[str] = ["CBRE", "JLL", "SPG"]

# ---------------------------------------------------------------------------
# spaCy & Sentence-Transformers
# ---------------------------------------------------------------------------
SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION: int = 384  # output dim for all-MiniLM-L6-v2
CHUNK_WORD_SIZE: int = int(os.getenv("CHUNK_WORD_SIZE", "300"))

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "cre_intelligence")

# ---------------------------------------------------------------------------
# LLM API (OpenAI / Gemini — used *only* for final RAG synthesis)
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
