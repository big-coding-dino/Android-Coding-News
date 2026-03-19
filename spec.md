ndroid Weekly RAG — Backend Spec

## Overview

A Python backend that ingests Android Weekly newsletter articles into a vector database and exposes a FastAPI server for search, listing, and AI chat. The client app (to be built separately) will consume this API.

---

## Tech Stack

| Concern | Choice | Reason |
|---|---|---|
| Ingestion orchestration | LlamaIndex | Handles chunking, embedding, storage |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace, local) | Free, no API key, good quality |
| Vector DB | ChromaDB (local) | Zero-config, persists to disk, easy to swap later |
| Article text extraction | `trafilatura` | Best-in-class boilerplate removal |
| YouTube transcripts | `youtube-transcript-api` | Free, no auth needed |
| GitHub READMEs | GitHub REST API (unauthenticated) | Free for public repos |
| LLM (chat) | Claude via Anthropic SDK | For query synthesis only |
| API server | FastAPI | Fast, async, auto-generates OpenAPI docs |

---

## Project Structure

```
backend/
├── data/
│   └── android_weekly_classified.json   # Pre-classified article index (already built)
├── storage/                             # ChromaDB + LlamaIndex persisted index (gitignored)
├── ingest.py                            # Run manually to ingest articles
├── server.py                            # FastAPI app
├── fetchers.py                          # fetch_content() per strategy
├── config.py                            # Settings, constants, API keys
└── requirements.txt
```

---

## Config (`config.py`)

```python
ANTHROPIC_API_KEY = "..."          # For chat endpoint only
DATA_FILE = "data/android_weekly_classified.json"
STORAGE_DIR = "./storage"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "claude-sonnet-4-6"
TOP_K = 8                          # How many chunks to retrieve per query

# Domains to skip (Medium = paywall headache)
SKIP_DOMAIN_TYPES = {"medium", "medium_pub"}
```

---

## Fetchers (`fetchers.py`)

Responsible for fetching full article text given an article entry from the JSON.

```python
def fetch_content(article: dict) -> str | None:
    ...
```

**Strategy routing:**

| `fetch_strategy` | Implementation |
|---|---|
| `trafilatura` | `trafilatura.fetch_url(url)` → `trafilatura.extract(html)` |
| `youtube_transcript` | Extract video ID from URL → `YouTubeTranscriptApi.get_transcript()` → join text |
| `github_api` | `GET https://api.github.com/repos/{owner}/{repo}/readme` with `Accept: application/vnd.github.raw` |
| `metadata_only` | Return `None` immediately |

**Fallback behaviour:** If any fetch returns `None` or raises, fall back to the Android Weekly curator description. Log the failure. Never crash the ingestion pipeline.

**Rate limiting:** Add a 0.5s delay between fetches to be polite. Trafilatura has its own timeout handling.

---

## Ingestion Script (`ingest.py`)

Run manually: `python ingest.py`

**Steps:**

1. Load `android_weekly_classified.json`
2. Filter out `SKIP_DOMAIN_TYPES`
3. For each article, call `fetch_content(article)`
4. Build a `LlamaIndex Document` with:
   - `text` = fetched content, or fallback to description
   - `metadata` = title, description, url, issue, date, category, domain\_type, has\_full\_content
5. Build a `VectorStoreIndex` backed by ChromaDB
6. Persist the index to `STORAGE_DIR`

**Re-run behaviour:** Check if an article URL already exists in the index before fetching. Skip if already ingested. This makes re-runs safe and incremental.

**LlamaIndex setup:**

```python
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.anthropic import Anthropic
import chromadb

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.llm = Anthropic(model=LLM_MODEL, api_key=ANTHROPIC_API_KEY)

chroma_client = chromadb.PersistentClient(path=STORAGE_DIR)
chroma_collection = chroma_client.get_or_create_collection("android_weekly")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)
```

---

## API Server (`server.py`)

Run: `uvicorn server:app --reload`

Loads the persisted index from `STORAGE_DIR` at startup. All endpoints are read-only.

---

### Endpoints

#### `GET /articles`

Returns the raw article list from the JSON (no vector search, no LLM). Used by the client to show the newsletter index — the list of articles per issue.

**Query params:**

| Param | Type | Description |
|---|---|---|
| `issue` | `int?` | Filter by issue number |
| `category` | `str?` | e.g. `"Articles & Tutorials"` |
| `domain_type` | `str?` | e.g. `"youtube"`, `"github"` |
| `q` | `str?` | Simple keyword filter on title + description |

**Response:**
```json
{
  "total": 142,
  "articles": [
    {
      "issue": 716,
      "date": "2026-03-01",
      "category": "Articles & Tutorials",
      "title": "Intro to Kotlin's Flow API",
      "description": "Dave Leeds shows how...",
      "url": "https://youtube.com/watch?v=...",
      "domain_type": "youtube"
    }
  ]
}
```

---

#### `GET /issues`

Returns a summary list of all available issues.

**Response:**
```json
{
  "issues": [
    { "issue": 716, "date": "2026-03-01", "article_count": 15 },
    { "issue": 715, "date": "2026-02-22", "article_count": 17 }
  ]
}
```

---

#### `GET /search`

Semantic search over article content using the vector index. Returns the most relevant articles for a natural language query.

**Query params:**

| Param | Type | Description |
|---|---|---|
| `q` | `str` | Required. Natural language query |
| `top_k` | `int?` | Number of results, default 8 |

**Response:**
```json
{
  "query": "how to use coroutines with viewmodel",
  "results": [
    {
      "score": 0.87,
      "title": "...",
      "description": "...",
      "url": "...",
      "issue": 710,
      "date": "2026-01-18",
      "category": "Articles & Tutorials",
      "has_full_content": true
    }
  ]
}
```

**Implementation note:** Use LlamaIndex's retriever directly (not the query engine) so you get raw chunks + scores without LLM synthesis:

```python
retriever = index.as_retriever(similarity_top_k=top_k)
nodes = retriever.retrieve(q)
```

---

#### `POST /chat`

RAG chat — takes a user question, retrieves relevant chunks, synthesises an answer with Claude.

**Request body:**
```json
{
  "message": "What are the best articles about Jetpack Compose performance?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

**Response:**
```json
{
  "answer": "Based on the articles in the newsletter...",
  "sources": [
    { "title": "...", "url": "...", "issue": 709 }
  ]
}
```

**Implementation note:** Use LlamaIndex's `CondenseQuestionChatEngine` or a simple `chat_engine` with memory. Sources come from the retrieved nodes' metadata. Always include sources in the response so the client can link back to the originals.

---

#### `GET /health`

Simple health check. Returns 200 with `{ "status": "ok", "articles_indexed": 142 }`.

---

## Error Handling

- All endpoints return standard HTTP error codes with a JSON body: `{ "error": "message" }`
- If the index fails to load at startup, the server should fail fast with a clear message (not silently serve broken responses)
- Fetch failures during ingestion are logged but never crash the pipeline

---

## Dependencies (`requirements.txt`)

```
llama-index
llama-index-embeddings-huggingface
llama-index-vector-stores-chroma
llama-index-llms-anthropic
chromadb
trafilatura
youtube-transcript-api
requests
fastapi
uvicorn
python-dotenv
```

---

## What's Out of Scope (for now)

- Cron job / scheduled re-ingestion
- Authentication on the API
- Kotlin Weekly (same approach, add later)
- Issue 714 is missing from the dataset (fetch failed) — investigate separately
- Streaming responses on `/chat`
- Full-text search (keyword, not semantic) — ChromaDB supports this later if needed

