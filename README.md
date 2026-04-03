
# CRE-Intelligence Pipeline

An end-to-end, AI-powered Commercial Real Estate (CRE) data intelligence pipeline and dashboard. This application ingests real-time RSS feeds, scrapes web insights, pulls financial API data, and merges it with static CSV/XLSX datasets. It structures this data using a unified Pydantic schema, enriches it via an automated Google Gemini LLM processing pipeline, embeds the documents into a Qdrant Vector Database, and surfaces insights through an interactive Streamlit dashboard.

## 🚀 Key Features

*   **Multi-Source Ingestion:** Seamlessly merges completely different formats into one unified schema.
*   **LLM Batch-Enrichment:** Automatically summarizes documents, classifies market sectors, and extracts complex Named Entities (Organizations & Geopolitical Entities) using Google Gemini.
*   **Semantic AI Search (RAG):** Answers user queries natively out of the dashboard by retrieving relevant vector documents and synthesizing exact grounded answers.
*   **Pre-Calculated Insights:** The intelligence engine crosses numerical Lending Data automatically against organic News Editorial Data to find correlating trends.
*   **Production Ready:** Fully dockerized with caching setups and production-grade REST APIs for instantaneous scaling.

## 🛠️ Technology Stack

### Core Technologies
*   **Python 3.11:** The backbone of the entire data engineering and machine learning pipeline.
*   **Docker & Docker Compose:** Used to containerize the entire architecture, managing the Streamlit app layer and the Qdrant Vector DB networking cleanly.

### Data Ingestion & Parsing
*   **BeautifulSoup4 / LXML:** Headless scraping pipeline specifically tailored for parsing complex target elements from JLL and Altus Group market insights headers.
*   **Feedparser:** Real-time XML/RSS parser handling the Financial Times commercial property feed.
*   **Requests & urllib:** Integrated with SSL cert-bypasses for extremely bulletproof organic web requests.

### Data Processing & AI
*   **Sentence-Transformers (`all-MiniLM-L6-v2`):** Used to generate dense NLP vector embeddings off the text chunks locally and cheaply without network latency.
*   **Google Gemini (`gemini-1.5-flash`):** Replaced legacy NLP frameworks. Processes bulk batches of text locally to generate classifications, summaries, and perform complex Named-Entity Recognition (NER). Also acts as the synthesis AI for the Semantic Query Engine.
*   **Pydantic (v2):** Enforces a strict, normalized schema (`NormalisedDocument`) mapping all fields across 6 varying data sources.

### Database
*   **Qdrant Vector Database:** A brutally fast, rust-based vector database. This application utilizes the Qdrant Python Client to spin up a local instance mapping cosine measurements for document retrieval (`.query_points`).

### Frontend Visualization
*   **Streamlit:** Pure-Python frontend rendering framework displaying metrics, data tables, Plotly visualizations, and the chat engine.
*   **Plotly Express & Graph_Objects:** Used to render interactive, aesthetically tuned graphs such as entity timelines and bar charts.

## ⚙️ Running Locally

1. Create a `.env` file from the root directory or inject them into your environment variables:
   ```env
   # Ensure you supply a valid Gemini API key
   GEMINI_API_KEY="your-gemini-key"
   # Stable FMP API key for fetching company profiles
   FMP_API_KEY="your-fmp-key"
   ```
2. Build and spin up the Docker application:
   ```bash
   docker compose down && docker compose up --build -d
   ```
3. Access the interactive web dashboard:
   - **Streamlit App:** `http://localhost:8501`
   - **Qdrant DB GUI (Optional):** `http://localhost:6333/dashboard`
