# Agentic RAG — IT Support Copilot

An enterprise IT support chat assistant built with **FastAPI**, **LangGraph**, and **Pinecone**. It answers questions from a private company knowledge base first, and only falls back to live web search (via Tavily) when that evidence is weak — grading and rewriting queries along the way. Every request's routing decisions are shown live as an "agent trace" in the UI and logged to a local SQLite audit log.

## How it works

The core is a LangGraph state machine ([app/rag/workflow.py](app/rag/workflow.py)):

```
START
  └─ route_question          → classify: "kb" (IT/support topic) or "direct" (greeting/chit-chat)
        ├─ direct_answer ────────────────────────────────────────────────────► END
        └─ retrieve_kb        → fetch top-k chunks from Pinecone
              └─ grade_kb      → LLM grades evidence as good/weak
                    ├─ good → generate_from_kb ─────────────────────────────► END
                    └─ weak → search_web       → Tavily web search
                                  └─ grade_web  → LLM grades evidence as good/weak
                                        ├─ good  → generate_from_web ───────► END
                                        ├─ weak & retries left → rewrite_query → back to retrieve_kb
                                        └─ weak & no retries   → insufficient ► END
```

Each node appends a human-readable step to `trace`, which the frontend renders in the "Agent Trace" panel so you can watch the routing/grading decisions as they happen.

## Tech stack

- **API**: FastAPI + Uvicorn, server-rendered UI via Jinja2 (no frontend build step — plain HTML/CSS/JS in [static/](static/) and [templates/](templates/))
- **Orchestration**: LangGraph
- **LLM**: Google Gemini (`langchain-google-genai`)
- **Web search fallback**: Tavily (`langchain-tavily`)
- **Vector store**: Pinecone (`langchain-pinecone`), serverless index created on demand
- **Embeddings**: HuggingFace sentence-transformers, local inference (`langchain-huggingface`)
- **Document ingestion**: PDF / TXT / Markdown / DOCX via `pypdf`, `python-docx`, `langchain-text-splitters`
- **Audit log**: SQLite ([app/services/audit.py](app/services/audit.py))

## Project structure

```
app/
  api/routes.py          # /api/health, /api/chat, /api/ingest
  core/config.py         # Settings (pydantic-settings, reads .env)
  core/logging.py        # Logging setup
  rag/state.py            # LangGraph state + structured-output schemas
  rag/workflow.py         # LangGraph nodes/edges (the agent itself)
  rag/vectorstore.py       # Pinecone index + embeddings + retriever
  services/ingestion.py    # File loading + chunking
  services/audit.py        # SQLite audit log
  main.py                  # FastAPI app, mounts routes/static/templates
static/                    # CSS/JS for the chat UI
templates/index.html       # Chat UI page
data/sample_kb/             # Sample IT handbook docs for demo ingestion
data/audit.db                # SQLite audit log (created at runtime)
uploads/                     # Uploaded documents land here before indexing
ingest_sample_kb.py          # One-off script: index data/sample_kb/ into Pinecone
run.py                       # Dev entrypoint (uvicorn with reload)
```

## Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini)
- A [Pinecone](https://www.pinecone.io/) API key
- A [Tavily](https://tavily.com/) API key (used only for the web-fallback path)

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
# then fill in your API keys
```

### Environment variables (`.env`)

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | yes | Google Gemini API key |
| `GEMINI_MODEL` | yes | Gemini model name, e.g. `gemini-2.5-flash` |
| `PINECONE_API_KEY` | yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | yes | Pinecone index name (created automatically if missing) |
| `PINECONE_NAMESPACE` | no | Pinecone namespace to use |
| `TAVILY_API_KEY` | yes, for web fallback | Tavily search API key — **note the exact spelling**, `API_KEY` with an underscore |
| `EMBEDDING_MODEL` | yes | HuggingFace embedding model, e.g. `sentence-transformers/all-mpnet-base-v2` |
| `EMBEDDING_DIMENSION` | yes | Must match the embedding model's output dimension (e.g. `768`) |
| `TOP_K` | no | Chunks retrieved per query (default `5`) |
| `MAX_RETRIES` | no | Query-rewrite retries before giving up (default `1`) |
| `ADMIN_API_KEY` | no | Key required in `X-Admin-Key` header to hit `/api/ingest` (default `password` — change this) |

> ⚠️ The Pinecone index is created (or recreated) with the dimension from `EMBEDDING_DIMENSION` the first time it's needed — make sure it matches `EMBEDDING_MODEL`'s actual output size, or vector inserts will fail.

## Running

```bash
python run.py
```

This starts Uvicorn with auto-reload at **http://127.0.0.1:8080**.

## Loading the sample knowledge base

To index the sample IT handbook docs in [data/sample_kb/](data/sample_kb/) into Pinecone:

```bash
python ingest_sample_kb.py
```

You can also upload your own PDF/TXT/MD/DOCX files from the UI ("+ Add Company Document") or directly via the API:

```bash
curl -X POST http://127.0.0.1:8080/api/ingest \
  -H "X-Admin-Key: <ADMIN_API_KEY>" \
  -F "file=@/path/to/document.pdf"
```

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/chat` | `{"question": "..."}` → runs the LangGraph agent, returns `answer`, `source_used`, `trace`, `citations` |
| `POST` | `/api/ingest` | Multipart file upload (requires `X-Admin-Key` header) → chunks and indexes the document into Pinecone |

## Audit log

Every `/api/chat` call is recorded in a local SQLite database at `data/audit.db` (question, source used, and the full trace as JSON) — see [app/services/audit.py](app/services/audit.py).
