---
id: "001"
title: "rag-foundation-specification"
stage: "spec"
date: "2025-12-20"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "rag-foundation"
branch: "001-rag-foundation"
user: "syedhuzaifa"
command: "/sp.specify"
labels: ["specification", "rag", "qdrant", "fastapi", "embeddings"]
links:
  spec: "specs/001-rag-foundation/spec.md"
  ticket: null
  adr: null
  pr: null
files:
  - "specs/001-rag-foundation/spec.md"
  - "specs/001-rag-foundation/checklists/requirements.md"
tests: []
prompt: |
  create the specification of a new feature

  **RAG Foundation: Embed Markdown docs → Qdrant → Retrieve**

  ---

  ## 0) Mandatory Constraints (Non-Negotiable)

  1. **Work ONLY inside the existing backend project**

     * ❌ Do NOT create a new `backend/` folder
     * ❌ Do NOT restructure the project
     * ✅ Add files only where specified below

  2. **Required environment variables are ALREADY added in `.env`**

     * `QDRANT_URL`
     * `QDRANT_API_KEY`
     * `FASTEMBED_MODEL`

     👉 **Do NOT modify `.env`**
     👉 **Assume these values exist and are valid**
     👉 Implementation must directly use them

  ---

  ## 1) Goal (Step-1 Only)

  Build a FastAPI backend that:

  * Reads **only `.md` files** from `book-source/docs/`
  * Chunks the markdown text
  * Generates embeddings using **FastEmbed**
  * Stores embeddings + payload in **Qdrant**
  * Retrieves **top-k relevant chunks** for a query

  ### Explicitly Out of Scope

  * ❌ OpenAI Agents SDK
  * ❌ `/chat` endpoint
  * ❌ Chat memory / persistence
  * ❌ UI or frontend work

  ---

  ## 2) Source Ingestion Rules

  * Source directory: `book-source/docs/`
  * File type: **ONLY `*.md`**
  * Recursive traversal: ✅ yes
  * Ignore directories:

    * `.git`
    * `node_modules`
    * `build`
    * `.docusaurus`
    * `__pycache__`

  ### Document Identity

  * `doc_id` = relative path from `book-source/docs/`

    * Example:

      ```
      MODULE-1/ROS2-ARCHITECTURE.md
      ```
  * `source` = same as `doc_id`

  ---

  ## 3) Environment Configuration (Already Done)

  The following **MUST be used as-is from `.env`**:

  * `QDRANT_URL` ✅ already set
  * `QDRANT_API_KEY` ✅ already set
  * `FASTEMBED_MODEL` ✅ already set

  Additional required env vars:

  * `QDRANT_COLLECTION=rag_chunks`
  * `CHUNK_SIZE=800`
  * `CHUNK_OVERLAP=120`
  * `DOCS_DIR=book-source/docs`

  ❌ No settings/config files
  ❌ No hardcoded values

  ---

  ## 4) Chunking Rules

  * Chunk by character length
  * Each chunk:

    * Size = `CHUNK_SIZE`
    * Overlap = `CHUNK_OVERLAP`
  * Each chunk must contain:

    * `chunk_id` (incremental integer)
    * `text` (string)

  ---

  ## 5) Qdrant Storage Requirements

  * Collection name: from `QDRANT_COLLECTION`
  * Vector field name: `embedding`

  ### Payload per chunk (MANDATORY)

  * `doc_id` (string)
  * `chunk_id` (int)
  * `source` (string)
  * `text` (string)

  ### Re-ingestion Rule

  * Re-ingesting the same document **must overwrite/upsert**
  * Stable point IDs must be derived from:

    ```
    doc_id + chunk_id
    ```

  ---

  ## 6) API Endpoints (Step-1)

  ### A) Health Check

  **GET `/health`**

  Response:

  ```json
  { "status": "ok" }
  ```

  Requirement:

  * Must verify Qdrant connectivity

  ---

  ### B) Ingest Markdown Directory

  **POST `/rag/ingest-dir`**

  Input:

  ```json
  { "wipe_collection": false }
  ```

  Output:

  ```json
  {
    "files_ingested": 25,
    "chunks_added": 640
  }
  ```

  Behavior:

  * Read all `.md` files from `DOCS_DIR`
  * Chunk → embed → upsert into Qdrant
  * If `wipe_collection=true`, clear/recreate collection first

  ---

  ### C) Retrieve Chunks

  **POST `/rag/retrieve`**

  Input:

  ```json
  {
    "query": "What is a ROS 2 node?",
    "top_k": 6,
    "filters": {
      "doc_id": "MODULE-1/ROS2-ARCHITECTURE.md"
    }
  }
  ```

  Output:

  ```json
  {
    "results": [
      {
        "score": 0.82,
        "doc_id": "MODULE-1/ROS2-ARCHITECTURE.md",
        "chunk_id": 12,
        "source": "MODULE-1/ROS2-ARCHITECTURE.md",
        "text": "..."
      }
    ]
  }
  ```

  Behavior:

  * Embed query using FastEmbed
  * Perform Qdrant similarity search
  * Return **chunks only** (no answer generation)

  ---

  ## 7) File Structure (INSIDE EXISTING BACKEND ONLY)

  Add **ONLY** the following:

  ```
  backend/src/
  ├─ api/
  │  ├─ router.py        # include rag router here
  │  └─ rag.py           # RAG endpoints
  └─ rag/
     ├─ chunking.py
     ├─ embeddings.py
     ├─ qdrant.py
     ├─ loader.py        # loads ONLY .md files
     └─ service.py
  ```

  ❌ No new backend project
  ❌ No duplicate folders

  ---

  ## 8) Test Cases (MANDATORY)

  ### Unit Tests

  1. **Chunking**

     * Chunk size respects `CHUNK_SIZE`
     * Overlap respects `CHUNK_OVERLAP`

  2. **Loader**

     * Loads only `.md` files
     * Skips ignored directories

  ---

  ### Integration Test (Qdrant running)

  3. **End-to-End**

     * Call `/rag/ingest-dir`
     * Call `/rag/retrieve` with a known query
     * Assert:

       * `results.length > 0`
       * Each result contains:

         * `doc_id`
         * `chunk_id`
         * `source`
         * `text`

  ---

  ## 9) Definition of Done (Step-1 Complete)

  ✅ Embeds **only** `book-source/docs/*.md`
  ✅ Stores vectors in Qdrant
  ✅ Retrieves relevant chunks
  ✅ Uses existing backend + existing `.env`
  ✅ All tests pass
response: |
  Successfully created the specification for the RAG Foundation feature. The specification includes:

  1. Complete User Scenarios & Testing section with 3 prioritized user stories:
     - Ingest Markdown Documentation (P1)
     - Retrieve Relevant Document Chunks (P1)
     - Health Check System Status (P2)
     - Edge cases identified

  2. Complete Requirements section with:
     - 14 functional requirements (FR-001 to FR-014)
     - 4 key entities defined

  3. Complete Success Criteria section with:
     - 8 measurable outcomes (SC-001 to SC-008)

  4. Created quality checklist at specs/001-rag-foundation/checklists/requirements.md
  5. All validation items passed successfully
outcome: "Specification created successfully with all mandatory sections completed, ready for planning phase"
evaluation: "The specification meets all requirements and is ready for the next phase of development"
---