"""
AI Processing Module
====================

Lightweight text chunking and embedding powered by **sentence-transformers**.
Additionally features an LLM enrichment batch processor that uses Google Gemini 
to classify content, extract entities (GPE + ORG), and generate summaries.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sentence_transformers import SentenceTransformer

from src.config import CHUNK_WORD_SIZE, EMBEDDING_MODEL, GEMINI_API_KEY, LLM_MODEL
from src.normalization import NormalisedDocument

logger = logging.getLogger(__name__)
import re

# ═══════════════════════════════════════════════════════════════════════════
# Entity Registry (Fallback for when AI is OFF)
# ═══════════════════════════════════════════════════════════════════════════

CRE_FIRMS = [
    "OakNorth", "Barings", "Barings Real Estate", "LaSalle", "LaSalle Investment Management",
    "JLL", "Jones Lang LaSalle", "CBRE", "Savills", "Cushman & Wakefield", 
    "Knight Frank", "Avison Young", "Colliers", "BNP Paribas Real Estate",
    "Altus Group", "Property Week", "Oxford Properties", "Landsec", "British Land",
    "HSBC", "Barclays", "Lloyds", "NatWest", "Santander", "Blackstone", "Starwood",
    "Brookfield", "Hines", "M&G", "Legal & General", "Schroders", "Aberdeen",
    "Allianz", "Axa", "Generali", "PIMCO", "Apollo", "KKR", "Hancock", "Nuveen"
]


# ═══════════════════════════════════════════════════════════════════════════
# Global Model Singletons
# ═══════════════════════════════════════════════════════════════════════════

_embedder: SentenceTransformer | None = None

def get_embedding_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer, loading it on first call."""
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        logger.info("Loading embedding model '%s' …", EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

# ═══════════════════════════════════════════════════════════════════════════
# LLM Enrichment (Batch)
# ═══════════════════════════════════════════════════════════════════════════

def llm_enrich_batch(docs: list[NormalisedDocument]) -> None:
    """Enrich a batch of documents using Gemini API simultaneously.
    
    This function modifies the documents IN-PLACE, appending extracted
    entities, classifications, and summaries into their `metadata` attribute.
    
    Args:
        docs: A list of documents (max 5-10 recommended in one hit).
    """
    if not docs:
        return
        
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is missing. Skipping LLM enrichment for %d docs.", len(docs))
        return

    try:
        import google.generativeai as genai
        import json
        import time
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(LLM_MODEL)
        
        # Build the JSON payload to send in the prompt
        payload = []
        for doc in docs:
            payload.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "content": f"{doc.content[:2000]}" # Truncate to save tokens
            })
            
        prompt = f"""You are an expert commercial real estate AI.
I am providing a JSON array of documents. You must analyse each document and return a JSON array containing your analysis.

For each document, provide:
1. "doc_id": The exact doc_id from the input.
2. "classification": A 1-3 word category (e.g. "Residential Market", "CRE Lending", "Company Profile", "Market News").
3. "summary": A concise, 1-2 sentence summary of the document.
4. "entities_gpe": A list of extracted geographical entities (cities/countries) mentioned.
5. "entities_org": A list of extracted organisation entities (companies/firms) mentioned.

Respond ONLY with valid JSON array containing these 5 fields per object. Do not include markdown blocks like ```json

--- INPUT BATCH ({len(docs)} documents) ---
{json.dumps(payload, indent=2)}
"""
        
        max_retries = 3
        base_delay = 20  # Wait 20 seconds initially if rate limited (the error asks for 17.5s)
        response = None
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                break
            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "quota" in exc_str or "exhausted" in exc_str:
                    if attempt < max_retries - 1:
                        logger.warning("Gemini API Rate limit hit. Retrying in %d seconds... (Attempt %d/%d)", base_delay, attempt + 1, max_retries)
                        time.sleep(base_delay)
                        base_delay *= 2  # Exponential backoff
                        continue
                logger.error("Failed to generate content: %s", exc)
                raise exc

        if not response:
            return

        text_resp = response.text.strip()
        
        # Strip potential markdown formatting that Gemini sometimes ignores instructions on
        if text_resp.startswith("```json"):
            text_resp = text_resp[7:]
        if text_resp.startswith("```"):
            text_resp = text_resp[3:]
        if text_resp.endswith("```"):
            text_resp = text_resp[:-3]
            
        results = json.loads(text_resp.strip())
        
        # Merge back into documents
        for result in results:
            doc_id = result.get("doc_id")
            # Find matching doc
            target = next((d for d in docs if d.doc_id == doc_id), None)
            if target:
                target.metadata.classification = result.get("classification", "")
                target.metadata.summary = result.get("summary", "")
                target.metadata.entities_gpe = result.get("entities_gpe", [])
                target.metadata.entities_org = result.get("entities_org", [])
                
        logger.info("Successfully enriched batch of %d documents using Gemini", len(docs))
        
    except Exception as exc:
        logger.error("LLM Enrichment batch failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Location Normalisation
# ═══════════════════════════════════════════════════════════════════════════

def normalise_locations(
    extracted_gpes: list[str],
    cities_lookup: list[dict[str, str]],
) -> list[str]:
    """Cross-reference extracted GPE entities against cities.csv."""
    city_names: dict[str, str] = {}
    for row in cities_lookup:
        name = row.get("city") or row.get("City") or row.get(" City") or ""
        name = name.strip().strip('"')
        if name:
            city_names[name.lower()] = name

    normalised: set[str] = set()
    for gpe in extracted_gpes:
        gpe_clean = gpe.strip()
        canonical = city_names.get(gpe_clean.lower())
        if canonical:
            normalised.add(canonical)
        elif gpe_clean:
            # If not in our US-centric CSV, keep it as is (but title case)
            normalised.add(gpe_clean.title())

    return sorted(normalised)


def perform_keyword_extraction(
    text: str,
    cities_lookup: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    """Scan text for known cities and organisations (fallback)."""
    found_cities = set()
    found_orgs = set()
    text_lower = text.lower()

    # 1. Look for cities
    for row in cities_lookup:
        city = row.get("city") or row.get("City") or ""
        city = city.strip().strip('"')
        if city and len(city) > 3: # Avoid very short names matching accidentally
            # Use regex to match only full words
            pattern = rf"\b{re.escape(city.lower())}\b"
            if re.search(pattern, text_lower):
                found_cities.add(city)

    # 2. Look for organisations
    for org in CRE_FIRMS:
        if org and len(org) > 2:
            pattern = rf"\b{re.escape(org.lower())}\b"
            if re.search(pattern, text_lower):
                found_orgs.add(org)

    return sorted(found_cities), sorted(found_orgs)

# ═══════════════════════════════════════════════════════════════════════════
# Text Chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_WORD_SIZE) -> list[str]:
    """Split text into overlapping chunks of approximately *chunk_size* words."""
    words = text.split()
    if not words:
        return []

    overlap = max(1, chunk_size // 10)
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks

# ═══════════════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════════════

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Compute dense vector embeddings for a batch of text strings."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()

def embed_single(text: str) -> list[float]:
    """Embed a single text string."""
    return embed_texts([text])[0]

# ═══════════════════════════════════════════════════════════════════════════
# High-Level Processing Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def process_document(
    doc: NormalisedDocument,
    cities_lookup: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Process a single document into embedded chunk records."""
    full_text = f"{doc.title}. {doc.content}"
    
    # Combine AI-extracted GPEs with a keyword fallback to ensure data visibility
    fallback_cities, fallback_orgs = perform_keyword_extraction(full_text, cities_lookup)
    
    # Merge ORGs
    doc_orgs = set(doc.metadata.entities_org)
    doc_orgs.update(fallback_orgs)
    
    # Special: XLSX data already has Lender/Borrower in metadata
    lender = doc.metadata.extra.get("lender")
    borrower = doc.metadata.extra.get("borrower")
    if lender: doc_orgs.add(str(lender))
    if borrower: doc_orgs.add(str(borrower))
    
    doc.metadata.entities_org = sorted(list(doc_orgs))

    # Merge GPEs/Locations
    all_gpes = set(doc.metadata.entities_gpe)
    all_gpes.update(fallback_cities)
    doc.metadata.entities_gpe = sorted(list(all_gpes))

    # Locations are normalised from whatever GPEs were found (either by LLM or keyword)
    doc.metadata.locations = normalise_locations(doc.metadata.entities_gpe, cities_lookup)

    chunks = chunk_text(full_text)
    if not chunks:
        chunks = [full_text] if full_text.strip() else ["(empty)"]

    embeddings = embed_texts(chunks)

    records: list[dict[str, Any]] = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        records.append(
            {
                "doc_id": doc.doc_id,
                "chunk_id": f"{doc.doc_id}_chunk_{idx}",
                "chunk_index": idx,
                "chunk_text": chunk,
                "embedding": embedding,
                "source_type": doc.source_type,
                "title": doc.title,
                "published_date": doc.metadata.published_date,
                "source_url": doc.metadata.source_url,
                "classification": doc.metadata.classification,
                "summary": doc.metadata.summary,
                "locations": doc.metadata.locations,
                "entities_gpe": doc.metadata.entities_gpe,
                "entities_org": doc.metadata.entities_org,
                "extra": doc.metadata.extra,
            }
        )

    return records


def process_documents(
    docs: list[NormalisedDocument],
    cities_lookup: list[dict[str, str]],
    enrich_llm: bool = False,
) -> list[dict[str, Any]]:
    """Process a batch of normalised documents through the AI pipeline."""
    
    # ── 1. Batch LLM Enrichment ──
    # ONLY enrich if explicitly requested (to save Gemini API quota).
    if enrich_llm:
        enrichable_docs = [d for d in docs if d.source_type in {"rss", "scraping", "api"}]
        
        import time
        
        # Process in batches of 5 to avoid overloading the LLM output buffer
        batch_size = 5
        for i in range(0, len(enrichable_docs), batch_size):
            batch = enrichable_docs[i:i + batch_size]
            llm_enrich_batch(batch)
            
            # Add a delay between batches to respect free tier RPM limits (15 Requests/Min)
            if i + batch_size < len(enrichable_docs):
                logger.info("Waiting 4 seconds before next Gemini API batch to respect rate limits...")
                time.sleep(4)
    else:
        logger.info("Skipping LLM enrichment to preserve API quota.")

    # ── 2. Vector Chunking & Embedding ──
    all_records: list[dict[str, Any]] = []
    for doc in docs:
        try:
            all_records.extend(process_document(doc, cities_lookup))
        except Exception as exc:
            logger.error("Failed to process doc '%s': %s", doc.title, exc)
            
    logger.info("Processed %d documents → %d chunk records", len(docs), len(all_records))
    return all_records
