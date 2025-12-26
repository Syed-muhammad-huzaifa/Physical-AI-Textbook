---
id: "001"
title: "rag-foundation-planning"
stage: "plan"
date: "2025-12-20"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "rag-foundation"
branch: "001-rag-foundation"
user: "syedhuzaifa"
command: "/sp.plan"
labels: ["planning", "architecture", "rag", "qdrant", "fastapi", "embeddings"]
links:
  spec: "specs/001-rag-foundation/spec.md"
  plan: "specs/001-rag-foundation/plan.md"
  ticket: null
  adr: null
  pr: null
files:
  - "specs/001-rag-foundation/plan.md"
  - "specs/001-rag-foundation/research.md"
  - "specs/001-rag-foundation/data-model.md"
  - "specs/001-rag-foundation/quickstart.md"
  - "specs/001-rag-foundation/contracts/rag-api.yaml"
tests: []
prompt: |
  now create a plan  Plan (FastAPI + Qdrant + FastEmbed) — inside existing backend

  ### 1) Create the required files (no new backend folder)

  Add only:

  * `backend/src/api/rag.py`
  * `backend/src/rag/loader.py`
  * `backend/src/rag/chunking.py`
  * `backend/src/rag/embeddings.py`
  * `backend/src/rag/qdrant.py`
  * `backend/src/rag/service.py`

  ---

  ### 2) Implement `.md` loader (`rag/loader.py`)

  * Read `DOCS_DIR=book-source/docs`
  * Recursively collect **only** `*.md`
  * Skip ignored folders: `.git`, `node_modules`, `build`, `.docusaurus`, `__pycache__`
  * Return list of `(doc_id, source, text)` where:

    * `doc_id` = relative path from `book-source/docs/`
    * `source` = same as `doc_id`

  ---

  ### 3) Implement chunking (`rag/chunking.py`)

  * `chunk_text(text, chunk_size, overlap) -> list[{chunk_id, text}]`
  * Character-based chunks with overlap from `.env`:

    * `CHUNK_SIZE`
    * `CHUNK_OVERLAP`

  ---

  ### 4) Implement FastEmbed wrapper (`rag/embeddings.py`)

  * Load model from `.env` **(already set)**: `FASTEMBED_MODEL`
  * Functions:

    * `embed_documents(list[str]) -> list[list[float]]`
    * `embed_query(str) -> list[float]`

  ---

  ### 5) Implement Qdrant layer (`rag/qdrant.py`)

  * Read `.env` **(already set)**:

    * `QDRANT_URL`, `QDRANT_API_KEY`
  * Use collection from `.env`: `QDRANT_COLLECTION`
  * Implement:

    * `ensure_collection(vector_dim)`
    * `upsert_chunks(chunks_with_vectors_and_payload)`
    * `search(query_vector, top_k, filter_doc_id=None)`
  * Stable point IDs: hash/ID derived from `doc_id + chunk_id`

  ---

  ### 6) Orchestration service (`rag/service.py`)

  Implement:

  * `ingest_dir(wipe_collection: bool) -> {files_ingested, chunks_added}`

    * load md files → chunk → embed → ensure collection → upsert
  * `retrieve(query, top_k, filters) -> results`

    * embed query → qdrant search → return payload + score

  ---

  ### 7) API endpoints (`api/rag.py`)

  Expose:

  * `GET /health` (checks Qdrant)
  * `POST /rag/ingest-dir` with `{ wipe_collection: bool }`
  * `POST /rag/retrieve` with `{ query, top_k, filters{doc_id?} }`

  ---

  ### 8) Wire router (`api/router.py`)

  * Include `rag` router under prefix `/rag`
  * Keep existing routes untouched

  ---

  ### 9) Tests (must run)

  Create tests (wherever your backend keeps tests, e.g. `backend/tests/`):

  1. **Unit: chunking**

     * size + overlap assertions
  2. **Unit: loader**

     * only `.md` returned
     * ignored dirs skipped
  3. **Integration: ingest + retrieve (requires Qdrant running)**

     * call `/rag/ingest-dir`
     * call `/rag/retrieve`
     * assert `results > 0` and required fields exist

  ---

  ### 10) Run checklist (manual)

  * Start FastAPI
  * Call `/rag/ingest-dir`
  * Call `/rag/retrieve`
  * Run all tests and confirm pass
response: |
  Successfully created the implementation plan for the RAG Foundation feature. The plan includes:

  1. Complete implementation plan document (plan.md) with technical context, constitution check, and project structure
  2. Research document (research.md) with technology decisions and alternatives
  3. Data model document (data-model.md) with entity definitions and validation rules
  4. API contracts (contracts/rag-api.yaml) with OpenAPI specification
  5. Quickstart guide (quickstart.md) with setup and usage instructions
  6. Updated agent context with new technologies

  The plan follows the constitution requirements using FastAPI, Qdrant, and FastEmbed. It extends the existing backend structure without creating new projects. All required files and dependencies are specified, with proper integration with the existing authentication system.
outcome: "Planning phase completed successfully with all required documentation created"
evaluation: "The implementation plan is comprehensive and follows all constitution requirements. It provides clear guidance for the development team to implement the RAG Foundation feature."
---