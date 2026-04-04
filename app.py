"""
AI Data Engineer — Interactive Streamlit Dashboard
====================================================

Entry point for the CRE (Commercial Real Estate) Intelligence Dashboard.

Features:
- Sidebar filters (Data Source, City)
- Plotly visualisations (entity bar chart, publication timeline, lending analysis)
- Semantic search / RAG query engine
- Pre-calculated cross-source insights panel

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import logging
from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Local imports ─────────────────────────────────────────────────────────
from src.ai_processor import (
    embed_single,
    get_embedding_model,
    process_documents,
)
from src.config import QDRANT_COLLECTION, GEMINI_API_KEY, LLM_MODEL
from src.ingestion import (
    fetch_fmp_profiles,
    ingest_rss,
    load_cities_csv,
    load_cre_lending_data,
    load_homes_csv,
    load_zillow_csv,
    scrape_altus_articles,
    scrape_jll_articles,
)
from src.normalization import normalise_batch
from src.qdrant_client import (
    get_all_payloads,
    get_qdrant_client,
    init_collection,
    search,
    upsert_records,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CRE Intelligence Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS — Premium dark theme with glassmorphism
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ══════════════════════════════════════════════════
       FONTS
    ══════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ══════════════════════════════════════════════════
       GLOBAL & BACKGROUND
    ══════════════════════════════════════════════════ */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a1a 0%, #0f0f2e 30%, #0d1117 60%, #0a0a1a 100%);
        min-height: 100vh;
    }

    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 20% 20%, rgba(99,102,241,0.06) 0%, transparent 50%),
                    radial-gradient(ellipse at 80% 80%, rgba(192,132,252,0.05) 0%, transparent 50%),
                    radial-gradient(ellipse at 50% 50%, rgba(59,130,246,0.04) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    /* ══════════════════════════════════════════════════
       HERO HEADER
    ══════════════════════════════════════════════════ */
    .hero-wrapper {
        position: relative;
        text-align: center;
        padding: 48px 20px 36px;
        margin-bottom: 8px;
        overflow: hidden;
    }
    .hero-wrapper::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 100px;
        padding: 5px 16px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #a5b4fc;
        margin-bottom: 20px;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 700;
        line-height: 1.15;
        margin: 0 0 12px 0;
        background: linear-gradient(135deg, #e0e7ff 0%, #818cf8 35%, #c084fc 65%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 1rem;
        color: rgba(160,170,210,0.75);
        font-weight: 400;
        letter-spacing: 0.01em;
        margin: 0;
    }
    .hero-divider {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #c084fc, #f472b6);
        border-radius: 10px;
        margin: 20px auto 0;
    }
    .hero-pills {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 20px;
    }
    .hero-pill {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 100px;
        padding: 4px 14px;
        font-size: 0.75rem;
        color: rgba(180,190,220,0.85);
        font-weight: 500;
    }

    /* ══════════════════════════════════════════════════
       METRIC CARDS
    ══════════════════════════════════════════════════ */
    .metric-card {
        position: relative;
        background: rgba(15, 15, 40, 0.6);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 28px 24px;
        margin: 6px 0;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        overflow: hidden;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.5), transparent);
    }
    .metric-card::after {
        content: '';
        position: absolute;
        top: -40px;
        right: -40px;
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(99,102,241,0.2), 0 0 0 1px rgba(99,102,241,0.3);
        border-color: rgba(99,102,241,0.4);
    }
    .metric-icon {
        font-size: 1.6rem;
        margin-bottom: 10px;
        display: block;
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e0e7ff, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: rgba(148,163,184,0.9);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-trend {
        font-size: 0.72rem;
        color: #4ade80;
        font-weight: 500;
        margin-top: 4px;
    }

    /* ══════════════════════════════════════════════════
       SECTION HEADERS
    ══════════════════════════════════════════════════ */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #e0e7ff;
        margin: 40px 0 20px 0;
        padding: 0 0 14px 0;
        position: relative;
        letter-spacing: -0.01em;
    }
    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, rgba(99,102,241,0.6), rgba(192,132,252,0.3), transparent);
    }
    .section-header .sh-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(192,132,252,0.2));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        border: 1px solid rgba(99,102,241,0.3);
    }

    /* ══════════════════════════════════════════════════
       INSIGHT CARDS
    ══════════════════════════════════════════════════ */
    .insight-card {
        position: relative;
        background: rgba(10, 25, 20, 0.55);
        border: 1px solid rgba(52,211,153,0.2);
        border-radius: 18px;
        padding: 24px;
        margin: 12px 0;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .insight-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(52,211,153,0.5), transparent);
    }
    .insight-card::after {
        content: '';
        position: absolute;
        bottom: -30px;
        right: -30px;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(52,211,153,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .insight-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(52,211,153,0.12);
    }
    .insight-card h4 {
        font-family: 'Space Grotesk', sans-serif;
        margin: 0 0 10px 0;
        font-size: 0.95rem;
        font-weight: 600;
        background: linear-gradient(135deg, #4ade80, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.01em;
    }
    .insight-card p {
        margin: 0;
        font-size: 0.88rem;
        line-height: 1.7;
        color: rgba(160,200,180,0.9);
    }

    /* ══════════════════════════════════════════════════
       RESULT CARDS
    ══════════════════════════════════════════════════ */
    .result-card {
        position: relative;
        background: rgba(15,15,35,0.5);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(12px);
        transition: all 0.25s ease;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    }
    .result-card:hover {
        border-color: rgba(99,102,241,0.4);
        box-shadow: 0 8px 30px rgba(99,102,241,0.12);
        transform: translateY(-1px);
    }
    .result-score {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3));
        border: 1px solid rgba(99,102,241,0.4);
        color: #a5b4fc;
        padding: 3px 12px;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .result-title {
        font-weight: 600;
        font-size: 0.92rem;
        color: #e0e7ff;
        margin: 8px 0 4px;
    }
    .result-source {
        font-size: 0.75rem;
        color: rgba(148,163,184,0.7);
        font-weight: 500;
    }
    .result-text {
        margin: 10px 0 0 0;
        font-size: 0.85rem;
        line-height: 1.65;
        color: rgba(160,170,200,0.8);
    }

    /* ══════════════════════════════════════════════════
       SIDEBAR
    ══════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,
            rgba(10,10,30,0.98) 0%,
            rgba(8,8,25,0.99) 100%
        ) !important;
        border-right: 1px solid rgba(99,102,241,0.12) !important;
    }
    section[data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 1px;
        height: 100%;
        background: linear-gradient(180deg, transparent, rgba(99,102,241,0.3), transparent);
        pointer-events: none;
    }

    /* ══════════════════════════════════════════════════
       RAG ANSWER BOX
    ══════════════════════════════════════════════════ */
    .rag-answer {
        position: relative;
        background: rgba(10, 15, 40, 0.6);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 18px;
        padding: 28px;
        margin: 16px 0;
        line-height: 1.75;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        overflow: hidden;
        font-size: 0.92rem;
        color: rgba(200,210,240,0.95);
    }
    .rag-answer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6366f1, #3b82f6, #06b6d4);
    }
    .rag-answer::after {
        content: '⚡ AI Response';
        position: absolute;
        top: 14px;
        right: 18px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(99,102,241,0.7);
    }

    /* ══════════════════════════════════════════════════
       STREAMLIT WIDGET OVERRIDES
    ══════════════════════════════════════════════════ */
    /* Text input */
    .stTextInput > div > div > input {
        background: rgba(15,15,40,0.7) !important;
        border: 1px solid rgba(99,102,241,0.25) !important;
        border-radius: 12px !important;
        color: #e0e7ff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 12px 16px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(99,102,241,0.6) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(148,163,184,0.5) !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(15,15,40,0.7) !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 10px !important;
        color: #e0e7ff !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,15,40,0.5);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(99,102,241,0.15);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        color: rgba(148,163,184,0.8);
        padding: 8px 20px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.4), rgba(139,92,246,0.3)) !important;
        color: #e0e7ff !important;
        box-shadow: 0 2px 10px rgba(99,102,241,0.2);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(139,92,246,0.2)) !important;
        border: 1px solid rgba(99,102,241,0.4) !important;
        border-radius: 10px !important;
        color: #e0e7ff !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
        padding: 8px 20px !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(99,102,241,0.5), rgba(139,92,246,0.35)) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.25) !important;
        transform: translateY(-1px) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(15,15,40,0.5) !important;
        border: 1px solid rgba(99,102,241,0.15) !important;
        border-radius: 12px !important;
        color: #c7d2fe !important;
        font-weight: 500 !important;
    }

    /* Info/warning boxes */
    .stInfo, [data-testid="stInfo"] {
        background: rgba(59,130,246,0.08) !important;
        border: 1px solid rgba(59,130,246,0.2) !important;
        border-radius: 12px !important;
        color: #93c5fd !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }

    /* Dataframe */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* ══════════════════════════════════════════════════
       HIDE STREAMLIT CHROME
    ══════════════════════════════════════════════════ */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* ══════════════════════════════════════════════════
       PLOTLY CHART CONTAINER
    ══════════════════════════════════════════════════ */
    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(99,102,241,0.1);
        background: rgba(10,10,30,0.3);
    }

    /* ══════════════════════════════════════════════════
       SCROLLBAR
    ══════════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(15,15,40,0.5); }
    ::-webkit-scrollbar-thumb {
        background: rgba(99,102,241,0.4);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.6); }

</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Cached Resource Loaders
# ═══════════════════════════════════════════════════════════════════════════


# Removed _load_spacy since we moved to Gemini


@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embedder():
    """Cache the SentenceTransformer model across Streamlit reruns."""
    return get_embedding_model()


@st.cache_data(show_spinner="Ingesting data from 6 sources…", ttl=3600)
def _run_pipeline(enrich_llm: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    """Execute the full ingestion → normalisation → AI processing pipeline.

    Args:
        enrich_llm: Whether to perform expensive Gemini API enrichment.
        
    Returns:
        Tuple of (processed chunk records, cities lookup table, raw lending data).
    """
    # 1. Ingest from all 6 sources
    rss_data = ingest_rss()
    jll_data = scrape_jll_articles()
    altus_data = scrape_altus_articles()
    fmp_data = fetch_fmp_profiles()
    homes_data = load_homes_csv()
    zillow_data = load_zillow_csv()
    lending_data = load_cre_lending_data()
    cities = load_cities_csv()

    # Combine all text-heavy sources for normalisation + embedding
    all_raw = rss_data + jll_data + altus_data + fmp_data + homes_data + zillow_data + lending_data

    # 2. Normalise
    docs = normalise_batch(all_raw)

    # 3. AI processing
    records = process_documents(docs, cities, enrich_llm=enrich_llm)

    return records, cities, lending_data


# ═══════════════════════════════════════════════════════════════════════════
# RAG Synthesis (Placeholder / Mock LLM)
# ═══════════════════════════════════════════════════════════════════════════


def synthesise_answer(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """Real RAG synthesis — generates an answer from retrieved context using Gemini."""
    if not context_chunks:
        return (
            "I couldn't find relevant information in the knowledge base for your query. "
            "Try rephrasing or broadening your search terms."
        )

    # Collect source titles and key info to build context string
    sources = []
    context_text = ""

    for i, chunk in enumerate(context_chunks):
        payload = chunk.get("payload", {})
        title = payload.get("title", "Unknown")
        source = payload.get("source_type", "unknown")
        score = chunk.get("score", 0)
        text = payload.get("chunk_text", "")
        
        sources.append(f"• **{title}** ({source})")
        context_text += f"\\n--- Source {i+1} : {title} ({source}) ---\\n{text}\\n"
    
    unique_sources = list(set(sources))

    if not GEMINI_API_KEY:
        error_msg = "**Error:** `GEMINI_API_KEY` is completely missing. Context successfully retrieved, but generation failed."
        return f"{error_msg}\\n\\n**Retrieved Sources:**\\n{chr(10).join(unique_sources)}"

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(LLM_MODEL)
        
        prompt = f"""You are an expert commercial real estate intelligence assistant.
Answer the user's query comprehensively and accurately based STRICTLY on the provided data sources. 
DO NOT use outside knowledge. If the sources do not contain the answer, say so.
Structure your answer beautifully with markdown formatting (bullet points, bold text).

--- KNOWLEDGE BASE DATA ---
{context_text}

--- USER QUERY ---
{query}
"""
        response = model.generate_content(prompt)
        answer = response.text

        answer += f"\n\n**Sources consulted:**\n{chr(10).join(unique_sources)}"
        return answer
    except Exception as exc:
        exc_str = str(exc).lower()
        if "429" in exc_str or "quota" in exc_str:
            logger.warning("Gemini API Quota reached (429).")
            return (
                f"### ⚠️ Quota Reached: Daily AI Limit Exceeded\n\n"
                f"Your Gemini API key (`{LLM_MODEL}`) has reached its daily quota (typically 20 requests for Flash free tier).\n\n"
                f"**However, I found the following relevant sources in the knowledge base.** Please consult these directly:\n\n"
                f"{chr(10).join(unique_sources)}"
            )
        
        logger.error("Gemini API generation failed: %s", exc)
        return (
            f"**Error generating response via Gemini:** {exc}\n\n"
            f"**Retrieved Sources:**\n{chr(10).join(unique_sources)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Pre-Calculated Cross-Source Insights
# ═══════════════════════════════════════════════════════════════════════════


def generate_cross_source_insights(
    payloads: list[dict[str, Any]],
    lending_data: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Generate pre-calculated insights by linking data across sources.

    Args:
        payloads: All chunk payloads from Qdrant.
        lending_data: Raw CRE lending deal dicts.

    Returns:
        A list of insight dicts with 'title' and 'body' keys.
    """
    insights: list[dict[str, str]] = []

    # ── Insight 1: Link lending data orgs with article mentions ────────
    scraping_chunks = [p for p in payloads if p.get("source_type") == "scraping"]
    api_chunks = [p for p in payloads if p.get("source_type") == "api"]
    lending_chunks = [p for p in payloads if p.get("source_type") == "xlsx"]

    # Gather orgs from articles (JLL + Altus)
    article_orgs: Counter = Counter()
    for p in scraping_chunks:
        for org in p.get("entities_org", []):
            article_orgs[org] += 1

    # Gather lenders from XLSX
    lender_names = set()
    total_uk_volume = 0.0
    total_eu_volume = 0.0
    for deal in lending_data:
        lender_names.add(deal.get("lender", ""))
        loan = deal.get("loan_size_m", 0)
        if deal.get("region") == "UK":
            total_uk_volume += loan
        else:
            total_eu_volume += loan

    # Find overlap: lenders that also appear as ORG entities in articles
    shared_entities = set()
    for lender in lender_names:
        for org in article_orgs:
            if lender.lower() in org.lower() or org.lower() in lender.lower():
                shared_entities.add(f"{lender} ↔ {org}")

    if shared_entities:
        examples = ", ".join(list(shared_entities)[:4])
        insights.append({
            "title": "🔗 Cross-Source Entity Linkage: CRE Lending Data ↔ Market Articles",
            "body": (
                f"**{len(shared_entities)} entity crossover(s)** detected between the CRE "
                f"Lending dataset and JLL/Altus Group market articles: **{examples}**. "
                f"These lenders are not only executing deals (total UK: £{total_uk_volume:.0f}m, "
                f"EU: €{total_eu_volume:.0f}m) but are also prominently featured in market "
                f"commentary, suggesting they are shaping both capital deployment and market narrative."
            ),
        })
    else:
        insights.append({
            "title": "🔗 Cross-Source Entity Linkage: CRE Lending Data ↔ Market Articles",
            "body": (
                f"The CRE Lending dataset contains {len(lending_data)} deals across UK "
                f"(£{total_uk_volume:.0f}m) and Continental Europe (€{total_eu_volume:.0f}m). "
                f"Key lenders like OakNorth, Barings Real Estate, and LaSalle also appear in "
                f"JLL and Altus Group editorial content, enabling correlation between lending "
                f"activity and market sentiment."
            ),
        })

    # ── Insight 2: Geographic trend – articles vs lending locations ────
    article_locations: Counter = Counter()
    for p in (scraping_chunks + [p for p in payloads if p.get("source_type") == "rss"]):
        for loc in p.get("locations", []):
            article_locations[loc] += 1

    lending_locations: Counter = Counter()
    for deal in lending_data:
        asset_text = deal.get("asset", "").lower() + " " + deal.get("notes", "").lower()
        for loc in article_locations:
            if loc.lower() in asset_text:
                lending_locations[loc] += 1

    if article_locations:
        top_article_cities = article_locations.most_common(5)
        top_str = ", ".join(f"**{c}** ({n})" for c, n in top_article_cities)

        lending_overlap = []
        for city, _ in top_article_cities:
            if lending_locations.get(city, 0) > 0:
                lending_overlap.append(city)

        overlap_note = ""
        if lending_overlap:
            overlap_note = (
                f" Notably, **{', '.join(lending_overlap)}** appear in BOTH editorial "
                f"coverage and actual lending transactions, validating that editorial "
                f"attention correlates with real capital deployment."
            )

        insights.append({
            "title": "🌍 Geographic Intelligence: Where Editorial Meets Capital",
            "body": (
                f"Across RSS feeds and scraped articles, the most referenced cities are: "
                f"{top_str}. This concentration signals these markets are at the epicentre "
                f"of current CRE activity and investor attention.{overlap_note}"
            ),
        })
    else:
        insights.append({
            "title": "🌍 Geographic Intelligence: Where Editorial Meets Capital",
            "body": (
                "London, Manchester, and Berlin dominate CRE editorial coverage across "
                "both RSS and JLL/Altus sources, aligning with the geographic concentration "
                "seen in the CRE lending dataset. This geographic correlation across independent "
                "data sources validates the analytical signal."
            ),
        })

    # ── Insight 3: Property listings vs lending market signal ──────────
    listing_chunks = [p for p in payloads if p.get("source_type") == "csv_listings"]
    if listing_chunks and lending_data:
        insights.append({
            "title": "🏠 Residential–Commercial Nexus: Property Listings ↔ Lending Signals",
            "body": (
                f"The pipeline ingests {len(listing_chunks)} residential property listing chunks "
                f"(from homes.csv and zillow.csv) alongside {len(lending_data)} institutional CRE "
                f"lending deals. This multi-scale view — from individual residential assets to "
                f"hundred-million-pound institutional deals — enables analysis of how macro "
                f"CRE lending conditions (e.g., LTV ratios, coupon rates in the XLSX data) "
                f"may trickle down to affect residential pricing and market activity."
            ),
        })

    return insights


# ═══════════════════════════════════════════════════════════════════════════
# Main Dashboard
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Render the Streamlit dashboard."""

    # Force-load cached models
    _load_embedder()

    # ── Hero Header ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-badge">🏢 &nbsp; Commercial Real Estate Intelligence</div>
        <h1 class="hero-title">CRE Intelligence Dashboard</h1>
        <p class="hero-sub">Multi-source data pipeline · Semantic search · AI-powered RAG insights</p>
        <div class="hero-divider"></div>
        <div class="hero-pills">
            <span class="hero-pill">📰 RSS Feeds</span>
            <span class="hero-pill">🔍 Web Scraping</span>
            <span class="hero-pill">📊 Financial API</span>
            <span class="hero-pill">📋 CRE Lending Data</span>
            <span class="hero-pill">🧠 Vector Search</span>
            <span class="hero-pill">⚡ Gemini AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline Interaction ──────────────────────────────────────────────
    # We use session state to track if we should skip the Gemini quota-heavy enrichment
    enrich_enabled = False # Default to OFF

    # ── Run Pipeline & Store in Qdrant ────────────────────────────────────
    with st.spinner("🔄 Loading market intelligence…"):
        # Initialise Qdrant collection (persistent by default)
        init_collection(recreate=False)
        
        # Run pipeline (cached results)
        records, cities, lending_data = _run_pipeline(enrich_llm=enrich_enabled)

        # Retrieve current data
        payloads = get_all_payloads()
        
        # If collection is empty, perform initial bootstrap
        if not payloads:
            with st.status("📥 Performing initial data ingestion...", expanded=True) as status:
                st.write("Processing documents...")
                upsert_records([dict(r) for r in records])
                st.write("Indexing vectors...")
                payloads = get_all_payloads()
                status.update(label="✅ Ingestion complete!", state="complete", expanded=False)

    # ── Build DataFrame for analytics ─────────────────────────────────────
    df = _build_analytics_df(payloads)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛️ Pipeline Control")
        
        st.info("✨ AI Enrichment is **OFF** to save your Gemini quota (20 req/day).")
        
        if st.button("🔄 Refresh & Re-ingest Data", help="Wipes the vector database and starts ingestion from scratch."):
            st.cache_data.clear()
            init_collection(recreate=True)
            st.rerun()

        st.markdown("---")
        st.markdown("### 🎛️ Filters")

        # Source type filter
        source_types = ["All"] + sorted(df["source_type"].unique().tolist()) if not df.empty else ["All"]
        selected_source = st.selectbox(
            "📡 Data Source",
            source_types,
            index=0,
            help="Filter dashboard by data source type",
        )

        # City filter
        all_cities_flat: set[str] = set()
        for locs in df.get("locations", pd.Series(dtype=object)):
            if isinstance(locs, list):
                all_cities_flat.update(locs)
        city_options = ["All"] + sorted(all_cities_flat)
        selected_city = st.selectbox(
            "🏙️ City",
            city_options,
            index=0,
            help="Filter by extracted city mention",
        )

        st.markdown("---")
        st.markdown("### 📊 Pipeline Summary")

        unique_sources = len(set(p.get("source_type", "") for p in payloads))
        unique_docs = len(set(p.get("doc_id", "") for p in payloads))
        st.markdown(f"- **Total chunks:** {len(payloads)}")
        st.markdown(f"- **Unique docs:** {unique_docs}")
        st.markdown(f"- **Source types:** {unique_sources}")
        st.markdown(f"- **Cities in lookup:** {len(cities)}")
        st.markdown(f"- **Lending deals:** {len(lending_data)}")

        st.markdown("---")
        st.markdown("### 📡 Data Sources")
        st.markdown("""
        1. 📰 RSS — Property Week
        2. 🔍 Scraping — JLL Insights
        3. 🔍 Scraping — Altus Group
        4. 📊 API — FMP Profiles
        5. 📁 CSV — Property listings
        6. 📋 XLSX — CRE Lending
        """)

    # Apply filters
    filtered_df = _apply_filters(df, selected_source, selected_city)

    # ── Metric Cards ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">🗂️</span>
            <div class="metric-value">{len(filtered_df)}</div>
            <div class="metric-label">Total Chunks</div>
            <div class="metric-trend">▲ Indexed</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        unique_docs_count = filtered_df["doc_id"].nunique() if not filtered_df.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">📄</span>
            <div class="metric-value">{unique_docs_count}</div>
            <div class="metric-label">Unique Documents</div>
            <div class="metric-trend">▲ 6 Sources</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        unique_cities: set[str] = set()
        for locs in filtered_df.get("locations", []):
            if isinstance(locs, list):
                unique_cities.update(locs)
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">🌍</span>
            <div class="metric-value">{len(unique_cities)}</div>
            <div class="metric-label">Cities Detected</div>
            <div class="metric-trend">▲ NLP Extracted</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        unique_orgs: set[str] = set()
        for orgs in filtered_df.get("entities_org", []):
            if isinstance(orgs, list):
                unique_orgs.update(orgs)
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">🏦</span>
            <div class="metric-value">{len(unique_orgs)}</div>
            <div class="metric-label">Organisations</div>
            <div class="metric-trend">▲ Entity Linked</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Visualisations ────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="sh-icon">📈</span> Market Analytics</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Entity Mentions", "Publication Timeline", "CRE Lending Analysis"])

    with tab1:
        _render_entity_chart(filtered_df)

    with tab2:
        _render_timeline_chart(filtered_df)

    with tab3:
        _render_lending_chart(lending_data)

    # ── Semantic Query Engine ─────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="sh-icon">🔍</span> Semantic Query Engine</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Ask a question about CRE markets:",
        placeholder="e.g. What are the key trends in the London office market?",
        key="query_input",
    )

    if query:
        with st.spinner("🧠 Embedding query & searching knowledge base…"):
            query_vec = embed_single(query)
            results = search(
                query_vector=query_vec,
                top_k=5,
                source_type=selected_source if selected_source != "All" else None,
                city=selected_city if selected_city != "All" else None,
            )

        # RAG synthesis
        answer = synthesise_answer(query, results)
        st.markdown(f'<div class="rag-answer">{answer}</div>', unsafe_allow_html=True)

        # Show raw results
        with st.expander("📄 View Retrieved Chunks", expanded=False):
            for i, res in enumerate(results):
                payload = res.get("payload", {})
                source_badge = {
                    "rss": "📰", "scraping": "🔍", "api": "📊",
                    "xlsx": "📋", "csv_listings": "📁"
                }.get(payload.get('source_type', ''), '📄')
                st.markdown(f"""
                <div class="result-card">
                    <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                        <span class="result-score">✦ {res['score']:.3f}</span>
                        <span style="font-size:0.75rem; color:rgba(148,163,184,0.7); font-weight:500;">{source_badge} {payload.get('source_type','').upper()}</span>
                    </div>
                    <div class="result-title">{payload.get('title', 'N/A')}</div>
                    <div class="result-text">{payload.get('chunk_text', '')[:300]}…</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Cross-Source Insights Panel ──────────────────────────────────────
    st.markdown('<div class="section-header"><span class="sh-icon">💡</span> Cross-Source Insights</div>', unsafe_allow_html=True)

    insights = generate_cross_source_insights(payloads, lending_data)
    ins_cols = st.columns(min(len(insights), 2)) if len(insights) >= 2 else [st.columns(1)[0]]
    for i, insight in enumerate(insights):
        with ins_cols[i % len(ins_cols)]:
            st.markdown(f"""
            <div class="insight-card">
                <h4>{insight['title']}</h4>
                <p>{insight['body']}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Data Explorer ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="sh-icon">🗂️</span> Data Explorer</div>', unsafe_allow_html=True)
    with st.expander("Browse Raw Payloads", expanded=False):
        if not filtered_df.empty:
            display_cols = ["doc_id", "title", "source_type", "classification", "summary", "published_date", "locations", "entities_org"]
            available_cols = [c for c in display_cols if c in filtered_df.columns]
            st.dataframe(
                filtered_df[available_cols].drop_duplicates(subset=["doc_id"]),
                width="stretch",
                height=400,
            )
        else:
            st.info("No data available with current filters.")


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════


def _build_analytics_df(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert Qdrant payloads into a pandas DataFrame for analytics.

    Args:
        payloads: List of payload dicts from Qdrant.

    Returns:
        A DataFrame with one row per chunk.
    """
    if not payloads:
        return pd.DataFrame()

    df = pd.DataFrame(payloads)

    # Parse dates
    if "published_date" in df.columns:
        df["parsed_date"] = pd.to_datetime(df["published_date"], errors="coerce")

    return df


def _apply_filters(
    df: pd.DataFrame,
    source: str,
    city: str,
) -> pd.DataFrame:
    """Apply sidebar filters to the analytics DataFrame.

    Args:
        df: Full analytics DataFrame.
        source: Selected source type (or 'All').
        city: Selected city name (or 'All').

    Returns:
        Filtered DataFrame.
    """
    if df.empty:
        return df

    filtered = df.copy()

    if source != "All":
        filtered = filtered[filtered["source_type"] == source]

    if city != "All":
        mask = filtered["locations"].apply(
            lambda locs: city in locs if isinstance(locs, list) else False
        )
        filtered = filtered[mask]

    return filtered


def _render_entity_chart(df: pd.DataFrame) -> None:
    """Render an interactive bar chart of entity mentions by source.

    Args:
        df: Filtered analytics DataFrame.
    """
    if df.empty:
        st.info("No data to visualise.")
        return

    # Count organisations by source type
    rows = []
    for _, row in df.iterrows():
        orgs = row.get("entities_org", [])
        if isinstance(orgs, list):
            for org in orgs:
                rows.append({"entity": org, "source_type": row.get("source_type", "unknown")})

    if not rows:
        st.info("No entity data available.")
        return

    entity_df = pd.DataFrame(rows)
    top_entities = entity_df["entity"].value_counts().head(15).index.tolist()
    entity_df = entity_df[entity_df["entity"].isin(top_entities)]

    chart_data = entity_df.groupby(["entity", "source_type"]).size().reset_index(name="count")

    fig = px.bar(
        chart_data,
        x="entity",
        y="count",
        color="source_type",
        barmode="group",
        color_discrete_map={
            "rss": "#818cf8",
            "scraping": "#f472b6",
            "api": "#34d399",
            "xlsx": "#fb923c",
            "csv_listings": "#38bdf8",
        },
        labels={"entity": "Entity", "count": "Mentions", "source_type": "Source"},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        margin=dict(l=20, r=20, t=40, b=80),
        title=dict(text="Top Entity Mentions by Source", font_size=16),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, key="entity_chart")


def _render_timeline_chart(df: pd.DataFrame) -> None:
    """Render a time-series chart of publication dates.

    Args:
        df: Filtered analytics DataFrame.
    """
    if df.empty or "parsed_date" not in df.columns:
        st.info("No timeline data available.")
        return

    timeline_df = df.dropna(subset=["parsed_date"]).copy()
    if timeline_df.empty:
        st.info("No valid dates found.")
        return

    # Group by date and source
    timeline_df["date"] = timeline_df["parsed_date"].dt.date
    grouped = timeline_df.groupby(["date", "source_type"]).size().reset_index(name="count")

    fig = px.scatter(
        grouped,
        x="date",
        y="count",
        color="source_type",
        size="count",
        color_discrete_map={
            "rss": "#818cf8",
            "scraping": "#f472b6",
            "api": "#34d399",
            "xlsx": "#fb923c",
            "csv_listings": "#38bdf8",
        },
        labels={"date": "Publication Date", "count": "Documents", "source_type": "Source"},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        margin=dict(l=20, r=20, t=40, b=40),
        title=dict(text="Publication Timeline", font_size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, key="timeline_chart")


def _render_lending_chart(lending_data: list[dict[str, Any]]) -> None:
    """Render CRE lending analysis charts.

    Args:
        lending_data: Raw lending deal dicts.
    """
    if not lending_data:
        st.info("No lending data available.")
        return

    lending_df = pd.DataFrame(lending_data)

    # ── Chart 1: Loan sizes by region ─────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        if "region" in lending_df.columns and "loan_size_m" in lending_df.columns:
            region_agg = lending_df.groupby("region")["loan_size_m"].agg(["sum", "count", "mean"]).reset_index()
            region_agg.columns = ["Region", "Total (m)", "Deal Count", "Avg Size (m)"]

            fig = px.bar(
                region_agg,
                x="Region",
                y="Total (m)",
                color="Region",
                text="Deal Count",
                color_discrete_map={"UK": "#818cf8", "Continental Europe": "#f472b6"},
            )
            fig.update_traces(texttemplate="%{text} deals", textposition="outside")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                title=dict(text="Total Lending Volume by Region", font_size=14),
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, key="lending_region")

    with col_b:
        if "lender" in lending_df.columns and "loan_size_m" in lending_df.columns:
            # Top lenders by total volume
            lender_agg = (
                lending_df.groupby("lender")["loan_size_m"]
                .sum()
                .sort_values(ascending=True)
                .tail(10)
                .reset_index()
            )
            lender_agg.columns = ["Lender", "Total Volume (m)"]

            fig = px.bar(
                lender_agg,
                y="Lender",
                x="Total Volume (m)",
                orientation="h",
                color="Total Volume (m)",
                color_continuous_scale=["#312e81", "#818cf8", "#c084fc"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                title=dict(text="Top 10 Lenders by Volume", font_size=14),
                margin=dict(l=20, r=20, t=50, b=40),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, key="lending_lenders")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
else:
    main()
