import re

with open("tests/test_ingestion.py", "r") as f:
    text = f.read()

# Fix imports
text = re.sub(r',\s*_get_sample_rss_data\s*', '', text)
text = re.sub(r',\s*_get_sample_jll_data\s*', '', text)
text = re.sub(r',\s*_get_sample_altus_data\s*', '', text)
text = re.sub(r',\s*_mock_fmp_profile\s*', '', text)

# Just delete test_ingest_rss_fallback
text = re.sub(r'@patch\("src\.ingestion\.feedparser\.parse"\)\ndef test_ingest_rss_fallback\(.*?\n\n', '', text, flags=re.DOTALL)

# And test_jll_fallback
text = re.sub(r'@patch\("src\.ingestion\.requests\.get"\)\ndef test_jll_scraping_fallback\(.*?\n\n', '', text, flags=re.DOTALL)

# And test_altus_fallback
text = re.sub(r'@patch\("src\.ingestion\.requests\.get"\)\ndef test_altus_scraping_fallback\(.*?\n\n', '', text, flags=re.DOTALL)

# And test_fmp_fallback
text = re.sub(r'@patch\("src\.ingestion\.requests\.get"\)\ndef test_fmp_fallback\(.*?\n\n', '', text, flags=re.DOTALL)

with open("tests/test_ingestion.py", "w") as f:
    f.write(text)

print("Tests patched.")
