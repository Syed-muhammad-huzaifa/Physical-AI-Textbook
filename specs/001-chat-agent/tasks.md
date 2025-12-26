# Implementation Tasks: Chat Agent with OpenAI SDK and Gemini

**Feature**: Chat Agent with OpenAI SDK and Gemini
**Branch**: `001-chat-agent` | **Date**: 2025-12-20
**Spec**: [specs/001-chat-agent/spec.md](specs/001-chat-agent/spec.md)

## Dependencies

User stories must be completed in priority order: US1 (P1) → US2 (P2) → US3 (P3). Each story builds on the foundational implementation created in Phase 2.

## Parallel Execution Examples

- T002, T003 can run in parallel during setup phase
- Within each user story phase, model/service/endpoint tasks can run in parallel if they don't depend on each other

## Implementation Strategy

- **MVP Scope**: US1 (RAG mode) only - minimal working implementation with basic query processing
- **Incremental Delivery**: Complete US1 → US2 → US3 in priority order
- **Test Early**: Each user story is independently testable per acceptance scenarios

---

## Phase 1: Setup Tasks

### Goal
Initialize project structure and install required dependencies for OpenAI Agents SDK and Gemini integration.

- [X] T001 Install OpenAI Agents SDK and Google Generative AI dependencies in backend
- [X] T002 [P] Verify Context7 MCP server is accessible and properly configured
- [X] T003 [P] Add GEMINI_API_KEY to environment configuration
- [X] T004 Create backend/src/api/chat_agent.py file structure with imports

---

## Phase 2: Foundational Tasks

### Goal
Implement core agent infrastructure including Context7 MCP consultation, system instructions, and the retrieve_chunks tool.

- [X] T005 Consult Context7 MCP for OpenAI Agents SDK usage patterns and tool schema
- [X] T006 Define system instructions for grounded responses and hallucination prevention in chat_agent.py
- [X] T007 Create retrieve_chunks tool that calls rag.service.retrieve() with proper parameters
- [X] T008 Implement conditional tool access based on mode (RAG vs Selection)
- [X] T009 Create agent using Gemini model with functional style (no classes)
- [X] T010 Implement agent input formatting for both RAG and Selection modes
- [X] T011 Create response schema validation for ChatResponse format
- [X] T012 Add proper error handling and validation for inputs

---

## Phase 3: [US1] Ask Questions Using RAG Mode

### Goal
Implement core RAG functionality allowing users to ask questions and get answers with citations from knowledge base.

**Independent Test**: Can be fully tested by sending a query in RAG mode and verifying that the response contains grounded answers with proper citations to source documents.

- [X] T013 [US1] Create POST /chat endpoint in chat_agent.py to handle RAG mode requests
- [X] T014 [US1] Extract query and mode from request, validate for RAG mode
- [X] T015 [US1] Format agent input as direct query string for RAG mode
- [X] T016 [US1] Run agent with retrieve_chunks tool available
- [X] T017 [US1] Parse agent output and extract citations from tool calls
- [X] T018 [US1] Return response with answer, citations array, and mode="rag"
- [X] T019 [US1] Test: Verify RAG mode returns grounded answers with citations
- [X] T020 [US1] Test: Verify filtered queries work with document filters

---

## Phase 4: [US2] Summarize Selected Text

### Goal
Implement selection mode allowing users to submit selected text with a query for analysis without knowledge base retrieval.

**Independent Test**: Can be fully tested by sending a query in selection mode with selected text and verifying that the agent responds appropriately to the provided content.

- [X] T021 [US2] Add mode detection logic to route to appropriate processing
- [X] T022 [US2] Format agent input with selected text and question format
- [X] T023 [US2] Run agent without retrieve_chunks tool (selection mode only)
- [X] T024 [US2] Return response with answer, empty citations array, and mode="selection"
- [X] T025 [US2] Test: Verify selection mode works with provided text only
- [X] T026 [US2] Test: Verify retrieval tool is not called in selection mode

---

## Phase 5: [US3] Handle Insufficient Information

### Goal
Implement proper handling of cases where the system cannot answer due to insufficient context.

**Independent Test**: Can be fully tested by sending queries that cannot be answered with available context and verifying that the system returns "Not enough information".

- [X] T027 [US3] Implement system instructions to return "Not enough information" when context is insufficient
- [X] T028 [US3] Test: Verify RAG mode returns "Not enough information" when no relevant chunks found
- [X] T029 [US3] Test: Verify selection mode returns "Not enough information" when selected text is insufficient
- [X] T030 [US3] Add proper fallback handling in agent processing

---

## Phase 6: Polish & Cross-Cutting Concerns

### Goal
Complete integration, testing, and ensure all requirements are met.

- [X] T031 Register /chat endpoint in backend/src/api/router.py
- [X] T032 Add comprehensive input validation for all request parameters
- [X] T033 Implement proper error responses for invalid inputs (400, 422 status codes)
- [X] T034 Add logging for debugging and monitoring
- [X] T035 Create integration tests covering all user stories
- [X] T036 Verify no class definitions exist in chat_agent.py (functional style only)
- [X] T037 Verify no persistence (no DB/files/chat history) is created
- [X] T038 Performance test: Ensure responses return within 10 seconds
- [X] T039 Update documentation with usage examples
- [X] T040 Final integration test: Verify all requirements from spec are met