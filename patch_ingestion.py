import re

with open("src/ingestion.py", "r") as f:
    text = f.read()

# 1. RSS
rss_try_block = """
def ingest_rss_feeds(
    rss_url: str = RSS_FEED_URL,
    max_items: int = RSS_MAX_ITEMS,
) -> list[dict[str, Any]]:
    # ... docstring ...
    logger.info("Ingesting RSS feed from %s", rss_url)
    results: list[dict[str, Any]] = []

    try:
        import ssl
        if hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context
            
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:max_items]:"""

# We'll replace the RSS function entirely with a regex to be safe
new_rss_func = """def ingest_rss_feeds(
    rss_url: str = RSS_FEED_URL,
    max_items: int = RSS_MAX_ITEMS,
) -> list[dict[str, Any]]:
    \"\"\"Ingest and normalise an RSS feed.\"\"\"
    logger.info("Ingesting RSS feed from %s", rss_url)
    results: list[dict[str, Any]] = []

    try:
        import ssl
        if hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context
            
        feed = feedparser.parse(rss_url)
        if feed.bozo and not feed.entries:
            logger.error("RSS feed error: %s", feed.bozo_exception)
            return []
            
        for entry in feed.entries[:max_items]:
            published = entry.get("published")
            if not published and entry.get("updated"):
                published = entry.get("updated")
            elif not published:
                published = datetime.now(timezone.utc).isoformat()

            results.append(
                {
                    "title": entry.get("title", "Untitled"),
                    "link": entry.get("link", rss_url),
                    "published_date": published,
                    "summary": entry.get("summary", ""),
                    "source_type": "rss",
                }
            )
        logger.info("Ingested %d items from RSS feed", len(results))
    except Exception as exc:
        logger.error("RSS ingestion failed: %s", exc)

    return results
"""
text = re.sub(r'def ingest_rss_feeds\(.*?(?=\n\n\ndef |# ══════════════)', new_rss_func + '\n\n', text, flags=re.DOTALL)

# Also remove _get_sample_rss_data
text = re.sub(r'def _get_sample_rss_data\(\) -> list\[dict\[str, Any\]\]:.*?(?=\n\n\n# ══════════════)', '', text, flags=re.DOTALL)

# 2. JLL Web Scraping
text = text.replace('a[href*=\'/trends-and-insights/\']', 'a[href*=\'/insights/\']')
text = re.sub(r'    # Fallback to sample data if scraping yields nothing\n    if not articles:\n        articles = _get_sample_jll_data\(\)\n', '', text)
text = re.sub(r'def _get_sample_jll_data\(\) -> list\[dict\[str, Any\]\]:.*?(?=\n\n\n# ══════════════)', '', text, flags=re.DOTALL)

# 3. Altus Web Scraping
text = re.sub(r'    # Fallback to sample data if scraping yields nothing\n    if not articles:\n        articles = _get_sample_altus_data\(\)\n', '', text)
text = re.sub(r'def _get_sample_altus_data\(\) -> list\[dict\[str, Any\]\]:.*?(?=\n\n\n# ══════════════)', '', text, flags=re.DOTALL)

# 4. FMP fetch profile
text = re.sub(r'            else:\n                profiles\.append\(_mock_fmp_profile\(ticker\)\)\n        except Exception as exc:\n            logger\.warning\("FMP API call failed for %s: %s — using mock", ticker, exc\)\n            profiles\.append\(_mock_fmp_profile\(ticker\)\)\n', 
              '            else:\n                logger.warning("FMP API returned invalid or empty data for %s", ticker)\n        except Exception as exc:\n            logger.warning("FMP API call failed for %s: %s", ticker, exc)\n', text)
text = re.sub(r'def _mock_fmp_profile\(ticker: str\) -> dict\[str, Any\]:.*?(?=\n\n\n# ══════════════|def )', '', text, flags=re.DOTALL)

# Clean up trailing spaces
text = '\n'.join([line.rstrip() for line in text.split('\n')]) + '\n'

with open("src/ingestion.py", "w") as f:
    f.write(text)

print("Patching successful.")
