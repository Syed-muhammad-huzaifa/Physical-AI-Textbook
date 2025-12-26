# Tasks: RAG Foundation - Embed Markdown docs → Qdrant → Retrieve

**Feature**: RAG Foundation - Embed Markdown docs → Qdrant → Retrieve
**Branch**: `001-rag-foundation`
**Created**: 2025-12-20
**Status**: Draft
**Input**: Implementation plan from `/specs/001-rag-foundation/plan.md`

## Implementation Strategy

This implementation will create a RAG (Retrieval Augmented Generation) system that reads markdown files from `book-source/docs/`, chunks them, generates embeddings using FastEmbed, stores them in Qdrant vector database, and provides retrieval endpoints. The system will be built as a FastAPI backend with dedicated modules for loading, chunking, embeddings, Qdrant integration, and service orchestration.

**MVP Scope**: User Story 1 (Ingest Markdown Documentation) - Basic ingestion functionality with health check

## Dependencies

- User Story 2 (Retrieve) depends on User Story 1 (Ingest) - documents must be ingested before they can be retrieved
- User Story 1 (Ingest) depends on all foundational components (dependencies, models, services)
- User Story 3 (Health Check) can be implemented in parallel with other stories

## Parallel Execution Examples

- T001-T007 (Setup) can run in parallel with environment configuration
- T008-T015 (Foundational) can be developed in parallel by different developers
- T016-T025 (US1 - Ingest) can run in parallel with T026-T035 (US2 - Retrieve) after foundational components are complete
- T036-T040 (US3 - Health Check) can run in parallel with other user stories

---

## Phase 1: Setup

### Goal
Initialize the project structure and install required dependencies

### Independent Test Criteria
- Project directory structure matches plan
- Dependencies can be installed successfully
- Environment variables are properly configured

### Tasks

- [x] T001 Create rag module directory structure at backend/src/rag/
- [x] T002 [P] Add FastEmbed dependency to backend/pyproject.toml: fastembed>=0.3.1
- [x] T003 [P] Add Qdrant client dependency to backend/pyproject.toml: qdrant-client>=1.9.0
- [x] T004 [P] Add pytest dependency to backend/pyproject.toml for testing: pytest>=8.0.0
- [x] T005 Create test directory structure at backend/tests/unit/
- [x] T006 Create test directory structure at backend/tests/integration/
- [x] T007 Create test directory structure at backend/tests/contract/

---

## Phase 2: Foundational Components

### Goal
Implement core components that all user stories depend on

### Independent Test Criteria
- All foundational components can be imported and used independently
- Environment variables are properly loaded and validated
- Core data models work as expected

### Tasks

- [x] T008 Create environment configuration at backend/src/rag/config.py
- [x] T009 [P] Create document chunk model at backend/src/rag/models.py
- [x] T010 [P] Create document model at backend/src/rag/models.py
- [x] T011 [P] Create query model at backend/src/rag/models.py
- [x] T012 [P] Create API request/response models at backend/src/rag/schemas.py
- [x] T013 [P] Create chunking utility at backend/src/rag/chunking.py
- [x] T014 [P] Create markdown loader at backend/src/rag/loader.py
- [x] T015 [P] Create embeddings wrapper at backend/src/rag/embeddings.py

---

## Phase 3: User Story 1 - Ingest Markdown Documentation (Priority: P1)

### Goal
Implement system to read markdown files from book-source/docs/, chunk them, generate embeddings, and store in Qdrant

### Independent Test Criteria
- Can call POST /rag/ingest-dir endpoint and verify markdown files are processed and stored in Qdrant with appropriate metadata
- Given directory with markdown files exists at book-source/docs/, when user calls POST /rag/ingest-dir, then all .md files are processed, chunked, embedded, and stored in Qdrant with appropriate metadata
- Given markdown files with large content exist, when user calls POST /rag/ingest-dir with wipe_collection=true, then collection is cleared and new chunks are stored with appropriate metadata

### Tests (if requested)

- [ ] T016 [P] [US1] Create unit test for chunking functionality at backend/tests/unit/test_chunking.py
- [ ] T017 [P] [US1] Create unit test for loader functionality at backend/tests/unit/test_loader.py

### Implementation

- [x] T018 [P] [US1] Implement Qdrant client wrapper at backend/src/rag/qdrant.py
- [x] T019 [P] [US1] Implement service orchestration logic at backend/src/rag/service.py
- [x] T020 [US1] Implement ingest_dir method in RAGService at backend/src/rag/service.py
- [x] T021 [US1] Create RAG API endpoints at backend/src/api/rag.py
- [x] T022 [US1] Implement POST /rag/ingest-dir endpoint with proper authentication
- [x] T023 [US1] Add validation for wipe_collection parameter in ingest endpoint
- [x] T024 [US1] Implement file processing logic in ingest_dir service method
- [x] T025 [US1] Test ingestion functionality with integration test at backend/tests/integration/test_rag_integration.py

---

## Phase 4: User Story 2 - Retrieve Relevant Document Chunks (Priority: P1)

### Goal
Implement system to search for relevant information from ingested documentation by providing a query and returning relevant chunks with metadata

### Independent Test Criteria
- Can call POST /rag/retrieve endpoint with a query and verify that relevant chunks are returned with proper metadata
- Given documents have been ingested into Qdrant, when user calls POST /rag/retrieve with a query, then the top-k most relevant chunks are returned with doc_id, chunk_id, source, and text
- Given documents have been ingested into Qdrant, when user calls POST /rag/retrieve with a query and filters, then only chunks matching the filters are returned

### Tests (if requested)

- [x] T026 [P] [US2] Create unit test for embedding query functionality at backend/tests/unit/test_embeddings.py

### Implementation

- [x] T027 [P] [US2] Implement retrieve method in RAGService at backend/src/rag/service.py
- [x] T028 [US2] Implement POST /rag/retrieve endpoint with proper authentication
- [x] T029 [US2] Add validation for query parameters (top_k, filters) in retrieve endpoint
- [x] T030 [US2] Implement query embedding logic in embeddings wrapper
- [x] T031 [US2] Implement search functionality in Qdrant client wrapper
- [x] T032 [US2] Add filtering support in Qdrant search method
- [x] T033 [US2] Test retrieval functionality with integration test at backend/tests/integration/test_rag_integration.py
- [x] T034 [US2] Create API contract test for retrieve endpoint at backend/tests/contract/test_rag_contracts.py
- [x] T035 [US2] Test end-to-end retrieval with known queries at backend/tests/integration/test_rag_integration.py

---

## Phase 5: User Story 3 - Health Check System Status (Priority: P2)

### Goal
Implement endpoint to verify that the RAG system is operational and can connect to its dependencies, particularly Qdrant

### Independent Test Criteria
- Can call GET /health endpoint and verify that the system reports its operational status
- Given system is running and connected to Qdrant, when user calls GET /health, then system returns status: "ok"

### Tests (if requested)

- [x] T036 [US3] Create health check test at backend/tests/unit/test_health.py

### Implementation

- [x] T037 [US3] Implement GET /health endpoint in backend/src/api/rag.py
- [x] T038 [US3] Add Qdrant connectivity check in health endpoint
- [x] T039 [US3] Test health check functionality with integration test
- [x] T040 [US3] Add health check to API router in backend/src/api/router.py

---

## Phase 6: Polish & Cross-Cutting Concerns

### Goal
Complete the implementation with proper error handling, documentation, and final integration

### Independent Test Criteria
- All endpoints work correctly with proper authentication
- Error conditions are handled gracefully
- System meets performance goals (500ms retrieval response time)
- All tests pass successfully

### Tasks

- [x] T041 Add comprehensive error handling for Qdrant connection failures
- [x] T042 Add validation for environment variables at startup
- [x] T043 Implement proper logging throughout RAG components
- [x] T044 Add request/response validation using Pydantic models
- [x] T045 Create documentation for API endpoints
- [x] T046 Perform end-to-end testing of all user stories
- [x] T047 Update README with RAG feature documentation
- [x] T048 Run all tests and fix any issues
- [x] T049 Optimize performance to meet 500ms retrieval goal
- [x] T050 Final integration test with complete workflow