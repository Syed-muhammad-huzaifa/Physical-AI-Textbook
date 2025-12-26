# Feature Specification: RAG Foundation - Embed Markdown docs → Qdrant → Retrieve

**Feature Branch**: `001-rag-foundation`
**Created**: 2025-12-20
**Status**: Draft
**Input**: User description: "create the specification of a new feature **RAG Foundation: Embed Markdown docs → Qdrant → Retrieve** - Build a FastAPI backend that reads only .md files from book-source/docs/, chunks the markdown text, generates embeddings using FastEmbed, stores embeddings + payload in Qdrant, and retrieves top-k relevant chunks for a query"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ingest Markdown Documentation (Priority: P1)

A system administrator or content manager needs to ingest markdown documentation files from the book-source/docs/ directory into the RAG system. The system reads all .md files recursively, chunks them, generates embeddings using FastEmbed, and stores them in Qdrant for later retrieval.

**Why this priority**: This is the foundational capability that enables the entire RAG system. Without the ability to ingest and store documents, no retrieval can occur.

**Independent Test**: Can be fully tested by calling the /rag/ingest-dir endpoint and verifying that markdown files are processed and stored in Qdrant with appropriate metadata.

**Acceptance Scenarios**:

1. **Given** a directory with markdown files exists at book-source/docs/, **When** the user calls POST /rag/ingest-dir, **Then** all .md files are processed, chunked, embedded, and stored in Qdrant with appropriate metadata
2. **Given** markdown files with large content exist, **When** the user calls POST /rag/ingest-dir with wipe_collection=true, **Then** the collection is cleared and new chunks are stored with appropriate metadata

---

### User Story 2 - Retrieve Relevant Document Chunks (Priority: P1)

A user or application needs to search for relevant information from the ingested documentation by providing a query. The system should return the most relevant chunks with their metadata to enable downstream applications to provide answers.

**Why this priority**: This is the core functionality that provides value to end users. Without retrieval, the ingestion process serves no purpose.

**Independent Test**: Can be fully tested by calling the /rag/retrieve endpoint with a query and verifying that relevant chunks are returned with proper metadata.

**Acceptance Scenarios**:

1. **Given** documents have been ingested into Qdrant, **When** the user calls POST /rag/retrieve with a query, **Then** the top-k most relevant chunks are returned with doc_id, chunk_id, source, and text
2. **Given** documents have been ingested into Qdrant, **When** the user calls POST /rag/retrieve with a query and filters, **Then** only chunks matching the filters are returned

---

### User Story 3 - Health Check System Status (Priority: P2)

A system administrator or monitoring service needs to verify that the RAG system is operational and can connect to its dependencies, particularly Qdrant.

**Why this priority**: Essential for operational reliability and monitoring. This ensures the system is healthy before processing requests.

**Independent Test**: Can be fully tested by calling the /health endpoint and verifying that the system reports its operational status.

**Acceptance Scenarios**:

1. **Given** the system is running and connected to Qdrant, **When** the user calls GET /health, **Then** the system returns status: "ok"

---

### Edge Cases

- What happens when Qdrant is unavailable during ingestion or retrieval?
- How does the system handle malformed markdown files?
- What happens when a document is re-ingested (should overwrite existing chunks)?
- How does the system handle extremely large documents that exceed memory limits?
- What happens when the query is empty or contains only special characters?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST read only .md files from the book-source/docs/ directory recursively
- **FR-002**: System MUST ignore specific directories (.git, node_modules, build, .docusaurus, __pycache__)
- **FR-003**: System MUST chunk markdown text by character length with configurable size (800) and overlap (120)
- **FR-004**: System MUST generate embeddings using FastEmbed technology
- **FR-005**: System MUST store embeddings and metadata in Qdrant vector database
- **FR-006**: System MUST provide a health check endpoint that verifies Qdrant connectivity
- **FR-007**: System MUST provide an ingestion endpoint that processes all markdown files and stores them in Qdrant
- **FR-008**: System MUST provide a retrieval endpoint that returns top-k relevant chunks for a given query
- **FR-009**: System MUST use environment variables for configuration (QDRANT_URL, QDRANT_API_KEY, FASTEMBED_MODEL, etc.)
- **FR-010**: System MUST support document identity using relative paths as doc_id
- **FR-011**: System MUST support re-ingestion that overwrites/upserts existing documents in Qdrant
- **FR-012**: System MUST return chunk metadata including doc_id, chunk_id, source, and text in retrieval results
- **FR-013**: System MUST support filtering by doc_id during retrieval operations
- **FR-014**: System MUST generate stable point IDs in Qdrant based on doc_id + chunk_id

### Key Entities

- **Document Chunk**: A segment of markdown text with associated metadata (doc_id, chunk_id, source, text) and embedding vector
- **Document**: A markdown file identified by its relative path from book-source/docs/ with associated chunks
- **Query**: A text input that is embedded and used to find similar chunks in the vector database
- **Qdrant Collection**: A container for storing document chunks with their embeddings and metadata

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The system successfully ingests all .md files from book-source/docs/ directory with 99% success rate
- **SC-002**: The system can retrieve relevant chunks in under 500ms for typical queries
- **SC-003**: The retrieval endpoint returns relevant results with 85% precision for test queries
- **SC-004**: The health check endpoint accurately reports system status within 100ms
- **SC-005**: The system can handle document re-ingestion without data corruption or duplication issues
- **SC-006**: The system supports configurable chunk size (default 800) and overlap (default 120) parameters
- **SC-007**: All required environment variables are properly utilized without hardcoded values
- **SC-008**: The system processes and stores at least 1000 document chunks without performance degradation