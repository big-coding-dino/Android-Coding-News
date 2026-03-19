# Android Weekly RAG — Progress Log

## What's been built

### Directory structure
```
/root/rag/
├── android_weekly_classified.json   # original data (copy also in backend/data/)
├── spec.md                          # full project spec
├── PROGRESS.md                      # this file
└── backend/
    ├── .env                         # OPENROUTER_API_KEY (gitignore this)
    ├── config.py                    # all settings and constants
    ├── requirements.txt             # pip dependencies
    ├── fetchers.py                  # fetch article content per strategy
    ├── ingest.py                    # builds the ChromaDB vector index
    ├── query.py                     # agentic RAG query engine (main logic)
    ├── data/
    │   └── android_weekly_classified.json
    └── storage/                     # ChromaDB persisted index (gitignore this)
```

---

## Config (`config.py`)

| Key | Value |
|---|---|
| `OPENROUTER_API_KEY` | loaded from `.env` |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | `stepfun/step-3.5-flash:free` (thinking model, 256K ctx, free) |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` (local HuggingFace, no API key needed) |
| `STORAGE_DIR` | `./storage` (ChromaDB persisted here) |
| `TOP_K` | `8` |
| `SKIP_DOMAIN_TYPES` | `{"medium", "medium_pub"}` (paywalled, skip) |

No `ANTHROPIC_API_KEY` — switched to OpenRouter (Anthropic account has no credits).

---

## Data

- **176 total articles** from issues 700–716 (Nov 2025 – Mar 2026)
- **40 skipped** (medium / medium_pub domain types)
- **136 ingested** → **337 chunks** in ChromaDB
- Fetch strategy breakdown:
  - `trafilatura`: 118 articles — full text extracted, ~3 returned 403 (fell back to description)
  - `github_api`: 16 articles — all succeeded, full README text
  - `youtube_transcript`: 41 articles — **deferred** (returns None, falls back to description)
  - `metadata_only`: 1 article — description only by design
- `has_full_content` metadata flag marks which chunks have real article text vs description fallback

---

## Ingestion (`ingest.py`)

- Run: `PYTHONPATH=/root/rag/backend /root/rag/.venv/bin/python3 ingest.py`
- Incremental: queries ChromaDB for existing URLs before fetching, skips already-ingested
- First run: `VectorStoreIndex.from_documents()`
- Re-run: `VectorStoreIndex.from_vector_store()` + `index.insert()` per new doc
- 0.5s delay between network fetches, no delay for `metadata_only`

---

## Query engine (`query.py`) — AGENTIC

The main logic. Three tools the LLM can call:

| Tool | What it does |
|---|---|
| `search_articles(query)` | Semantic search over ChromaDB vector index — always called first |
| `web_search(query)` | DuckDuckGo search for topics not in the newsletter |
| `fetch_url(url)` | Trafilatura fetch+extract for any URL the LLM wants to read in full |

**Flow:**
1. LLM always starts with `search_articles` (grounded in newsletter)
2. Reads results, decides if it needs more depth
3. Calls `web_search` and/or `fetch_url` as needed
4. Writes final answer citing both newsletter and web sources

**How to use:**
```python
from query import ask_agentic

# Single question
result = ask_agentic("how does Koin differ from Hilt?")
print(result["answer"])
print(result["sources"])      # {"newsletter": [...], "web": [...]}
print(result["tool_calls"])   # list of what the LLM called

# With chat history (multi-turn)
history = []
r1 = ask_agentic("tell me about Jetpack Compose performance", history=history)
history += [
    {"role": "user", "content": "tell me about Jetpack Compose performance"},
    {"role": "assistant", "content": r1["answer"]},
]
r2 = ask_agentic("which of those articles should I read first?", history=history)
```

**CLI:**
```bash
cd /root/rag/backend
PYTHONPATH=/root/rag/backend /root/rag/.venv/bin/python3 query.py "your question"
```

---

## Known issues / next tasks

### Immediate fixes needed in query.py
- `duckduckgo_search` package renamed to `ddgs` — shows RuntimeWarning. Fix: `pip install ddgs` and update import
- Web search query quality: searching "Hilt" returned Hilti power tools — system prompt needs stronger Android context injection into web queries (e.g. always append "Android development" to web_search queries)

### Not yet built
- `server.py` — FastAPI wrapping the query pipeline into HTTP endpoints:
  - `GET /articles` — list/filter raw articles from JSON
  - `GET /issues` — list all issues with article counts
  - `GET /search` — semantic search (non-agentic, fast)
  - `POST /chat` — agentic RAG chat (uses `ask_agentic` + history)
  - `GET /health` — status check

### Deferred features
- YouTube transcripts (`youtube_transcript` strategy) — 41 articles currently fall back to description only. Implementing would significantly improve quality for those articles. Needs `youtube-transcript-api` already installed.
- `.gitignore` — `storage/` and `.env` should be gitignored
- Server startup fail-fast — validate ChromaDB non-empty before serving

---

## Environment

- Python: 3.12 (system), venv at `/root/rag/.venv`
- Run anything with: `PYTHONPATH=/root/rag/backend /root/rag/.venv/bin/python3 <script>`
- All deps installed in venv via `requirements.txt`
- No pip available system-wide — always use `.venv/bin/pip`
