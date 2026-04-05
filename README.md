# Conversational AI Agent

Production-grade conversational AI agent with a Streamlit UI, LlamaIndex-based multi-agent workflows, and a smolagents tool-calling runner. This repository focuses on multimodal reasoning (text, images, audio, video), document-grounded answers, and configurable provider/model choices.

## What is implemented

### 1) LlamaIndex conversational agent (`llama_index_app/agent.py`)
- **Multi-agent workflow** (`ReActAgent` + `AgentWorkflow`) with a research agent and optional specialized agents (code execution, image analysis, media analysis, image generation/editing).
- **Modes**:
  - **API mode**: Gemini, OpenAI, or OpenRouter (selected per chat in the UI). Supports vector memory using provider embeddings; web search returns LLM-summarized answers from fetched pages.
  - **Local mode**: Qwen, Ministral, or GPT-OSS suites with specialized models (text, vision, audio/video, code, optional image generation/editing if diffusers are available).
- **Document handling**:
  - Docling for PDFs, Office files, and HTML.
  - CSV/Excel/JSON/text readers for structured data.
- **Web search**:
  - DDGS search + content extraction with retry/backoff.
  - Local mode: temporary in-memory `VectorStoreIndex` with hierarchical chunking and reranking (Jina or Qwen).
  - API mode: fetches page content from multiple results, then returns an LLM-summarized answer (no indexing).
- **Code execution** with a restricted global namespace.
- **Structured answer extraction** via Pydantic `LLMTextCompletionProgram`.

### 2) smolagents runner (`smolagents_app/agent.py`)
- **smolagents agent** built on `CodeAgent` with:
  - Web search (DuckDuckGo), page visiting, YouTube transcript, Python interpreter, and optional multimodal tool.
  - Optional MCP tool collections (filesystem, GitHub, Brave Search, Slack, Google Maps, Postgres, PubMed).
- Supports **Gemini** and **OpenAI** via `OpenAIServerModel`.
- Optional multimodal tool:
  - **Gemini**: images/audio/video via Gemini SDK.
  - **OpenAI**: images via Responses API; audio/video via transcription models.
- **Document loading** in memory (Docling-first). If Docling fails, the document is skipped.
- **Used for the Hugging Face GAIA challenge** as the final assignment for the AI Agent course; GAIA task files: documents are injected into the prompt via Docling; media files are passed as file paths with explicit tool instructions to call `multimodal_processor`.
- **Note**: Contains `llm_reformat()` function using Pydantic structured outputs for answer extraction, designed to comply with GAIA benchmark format constraints (number/string/comma-separated list formats).

### 3) Streamlit UI (`app.py`)
- Per-chat configuration (framework, mode, provider, model, and specialized features).
- Session persistence to `.chat_sessions`.
- LlamaIndex local mode: conversation stores with cached sources you can link into a chat.
- Prompt media uploaders for LlamaIndex (API + local) and smolagents (multimodal tool).
- API-mode document injection (full content inserted into prompt memory).

### 4) REST API (`api/app.py`)
- FastAPI backend that wraps both agent frameworks.
- Endpoints: `POST /sessions`, `GET /sessions`, `POST /sessions/{id}/chat`, `DELETE /sessions/{id}`, `GET /health`.
- Typed request/response schemas via Pydantic.
- Run with: `uvicorn api.app:app --reload --port 8000`.

### 5) Utilities
- `config.py`: Centralized Pydantic configuration — all environment variables validated at startup via `AppSettings`, typed `AgentConfig` model replacing untyped dicts.
- `llama_index_app/utils/vector_store.py`: Chroma-based conversation stores plus a shared library cache, with configurable embeddings (Jina, Qwen, Gemini, or OpenAI).
- `utils/session_manager.py`: session persistence (messages, metadata, agent config).
- `utils/retry.py`: retry-with-backoff decorator for resilient API calls.
- `llama_index_app/utils/document_processor.py`: Docling-driven file parsing + web/YT helpers.
- `smolagents_app/utils/mcp_connectors.py`: MCP server registration and loader helpers (scoped environment for subprocess security).

## Architecture

### Dual-framework design
LlamaIndex handles RAG-heavy orchestration with specialized sub-agents. smolagents handles tool-heavy code execution. They share no runtime state — each has its own model initialization, tools, and prompts.

### Model wrapper pattern
All local model wrappers in `llama_index_app/model_wrappers/` extend LlamaIndex's `CustomLLM`. They use lazy loading (`_ensure_hf()`) with thread-locked initialization to prevent GPU race conditions. Models are cached as singletons in `llama_index_app/custom_models.py` using double-checked locking. A framework-agnostic `ModelProvider` / `EmbeddingProvider` interface is defined in `llama_index_app/model_wrappers/base.py` for gradual decoupling.

### GPU memory management
`llama_index_app/model_wrappers/utils.py` provides `offload_rag_models()`, `offload_image_models()`, and `prepare_llm_for_inference()` to swap models between GPU and CPU. This is critical for running multiple specialized models on a single GPU (e.g., L4).

### RAG pipeline
- Embeddings: API mode uses provider-native embeddings (Gemini or OpenAI). Local mode defaults to Jina, configurable to Qwen.
- Vector store: ChromaDB, only used in LlamaIndex local mode.
- Chunking: `HierarchicalNodeParser` (2048 -> 512 -> 256) with leaf-node indexing.
- Two stores: per-session conversation store + shared library cache.

### UI decomposition
`app.py` is a thin orchestrator (~80 lines). All logic lives in `ui/` modules:
- `ui/state.py` — session state initialization
- `ui/sidebar.py` — session management, chat creation, settings
- `ui/chat.py` — chat interface, message handling, response generation
- `ui/documents.py` — document upload processing
- `ui/feedback.py` — Langfuse feedback controls
- `ui/vector_store.py` — vector store management UI

## How it works

### LlamaIndex local mode
- Uses **Jina** (default) or **Qwen** embeddings and **Chroma** conversation stores.
- Hierarchical chunking (`HierarchicalNodeParser`) and leaf-node indexing.
- Optional specialized agents for code, media analysis, image generation/editing.
- Chat uploads are indexed into Chroma and tracked in the shared library cache.

### LlamaIndex API mode
- Uses **Gemini/OpenAI** for reasoning, with provider-native embeddings.
- Document uploads can be indexed into vector memory for long-term retrieval.
- Web search fetches page content and returns an LLM-summarized answer (no indexing).

### smolagents
- CodeAgent-based assistant designed for tool-heavy tasks.
- API only; no local-mode execution.
- MCP tools can be enabled per chat.

## Best practices

- **Local multi-agent workflows** are GPU-intensive. For Qwen 30B + vision + code models, plan for **L4-class GPU or better**.
- **Qwen media analysis + image generation/editing** also requires an **L4-class GPU or better**.
- **GPT-OSS** can run image analysis + strong coding on **16 GB VRAM**.
- **Ministral** runs well on **16 GB VRAM** and offers native multimodal capabilities.
- **LlamaIndex local mode** is best for large document sets and offline workflows.
- **API mode** is best for quick setup and long-context analysis without local hardware requirements.

## Getting started

### Install dependencies
```
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Lock dependencies
```
uv pip compile requirements.txt -o requirements.lock
uv pip compile requirements.txt -o requirements.lock --upgrade
```

### Run the Streamlit app
```
streamlit run app.py
```
The UI opens at `http://localhost:8501`.

### Run the REST API
```
uvicorn api.app:app --reload --port 8000
```
API docs at `http://localhost:8000/docs`.

### Run with Docker
```
# CPU
docker compose up

# GPU (uncomment app-gpu service in docker-compose.yml)
docker compose up app-gpu
```

### Run tests
```
python -m pytest tests/ -v
```

## Using the Streamlit UI

### Create a chat
1. Click **New Chat** in the sidebar.
2. In **Settings**, provide at least one API key: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`.
3. Choose:
   - **Framework**: `llamaindex` or `smolagents`.
   - **Mode**: `API` or `Local` (smolagents requires API).
  - **Provider/Model** and specialized feature toggles.
  - If provider is **OpenRouter** (LlamaIndex API mode), the default model path is shown and can be edited in-app.
4. Click **Create Chat**.

### Upload documents
- **Local LlamaIndex**: documents are indexed into a per-chat vector store and cached in the shared library.
- **API mode**: documents can be indexed into vector memory (provider embeddings) for long-term retrieval.

### Multimodal attachments
- **LlamaIndex**: use **Prompt Media** to attach images/audio/video to the next prompt (local audio/video requires Media Analysis enabled).
- **smolagents**: use **Prompt media (multimodal)** to attach images/audio/video for the next prompt; the model is instructed to call `multimodal_processor`.
- PDFs and office docs should be uploaded via the document upload sections (chat files), not prompt media.

## Configuration

### Environment variables
All configuration is centralized in `config.py` using Pydantic settings. Variables are validated at startup.

**Core keys** (set at least one):
```
export GOOGLE_API_KEY="your-google-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

**Langfuse**:
```
export LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
export LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

**MCP servers (optional)**:
```
export GITHUB_PERSONAL_ACCESS_TOKEN="your-github-token"
export BRAVE_API_KEY="your-brave-api-key"
export SLACK_BOT_TOKEN="your-slack-bot-token"
export POSTGRES_CONNECTION_STRING="your-postgres-uri"
```

**OpenAI transcription model (optional)**:
```
export OPENAI_TRANSCRIBE_MODEL="gpt-4o-mini-transcribe"
```

**Weave tracing (optional, opt-in)**:
```
export WEAVE_ENABLED="true"
```
By default Weave initialization is disabled to avoid interactive Weights & Biases login prompts during app startup.

### Streamlit settings
Configured in `.streamlit/config.toml` (port, theme, uploads, logging).

## Repository structure

```
app.py                              # Streamlit entry point (thin orchestrator)
config.py                           # Centralized Pydantic configuration
ui/                                 # Streamlit UI modules
  state.py                          # Session state initialization
  sidebar.py                        # Session management, chat config, settings
  chat.py                           # Chat interface and response generation
  documents.py                      # Document upload handling
  feedback.py                       # Langfuse feedback controls
  vector_store.py                   # Vector store management UI
api/                                # FastAPI REST backend
  app.py                            # Routes and agent management
  schemas.py                        # Request/response Pydantic models
llama_index_app/
  agent.py                          # LlamaIndex multi-agent runtime
  custom_models.py                  # Cached model singletons
  models.py                         # Model initialization & global config
  model_wrappers/                   # CustomLLM implementations
    base.py                         # Framework-agnostic ModelProvider interface
    ...                             # Qwen, Gemini, OpenAI, Ministral, Jina wrappers
  utils/
    vector_store.py                 # ChromaDB + Jina embeddings
    document_processor.py           # Docling-based file parsing
smolagents_app/
  agent.py                          # smolagents CodeAgent (GAIA)
  utils/mcp_connectors.py           # MCP server definitions and loader
utils/
  session_manager.py                # Chat session persistence
  retry.py                          # Retry with exponential backoff
tests/                              # pytest test suite
  test_config.py                    # Configuration validation tests
  test_session_manager.py           # Session persistence tests
  test_api.py                       # FastAPI endpoint tests
  test_retry.py                     # Retry utility tests
.github/workflows/ci.yml           # CI pipeline (lint + test)
Dockerfile                          # CPU container
Dockerfile.gpu                      # GPU container (CUDA 12.6)
docker-compose.yml                  # Container orchestration
```

## Development

### Git conventions
This project uses [conventional commits](https://www.conventionalcommits.org/). See `.gitmessage` for the template. Types: `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, `chore`.

### Pre-commit hooks
```
pip install pre-commit
pre-commit install
```
Runs ruff linting and formatting on every commit.

### CI pipeline
GitHub Actions runs on every push/PR to `main`:
- **Lint**: ruff check + format
- **Test**: pytest with dummy API keys

## Notes
- **Embeddings vary by mode**: API mode uses the provider's own embeddings (Gemini or OpenAI). Local mode defaults to Jina (configurable to Qwen).
- **Vector stores are available for LlamaIndex local and API modes**.
- **smolagents has no local mode**.
- **Model offloading order matters**: always offload RAG models before loading the main LLM for inference.

## Demo data
See the `demo/` folder for sample chat session JSON files (`demo/chat_sessions_demo/`) and UI screenshots (`demo/UI_screenshots/`) useful for quick testing and reference.

## Troubleshooting

- **Model errors**: confirm GPU memory and correct model IDs.
- **API failures**: verify the API keys are set. Check `GET /health` for provider status.
- **Docling errors**: validate file type and ensure docling dependencies are installed.
- **MCP tools not loading**: verify required environment variables for each server.
- **Configuration errors at startup**: check `config.py` — Pydantic will report which env vars are invalid.
