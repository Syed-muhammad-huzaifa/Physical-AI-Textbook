# Feature Specification: Chat Agent with OpenAI SDK and Gemini

**Feature Branch**: `001-chat-agent`
**Created**: 2025-12-20
**Status**: Draft
**Input**: User description: "Create a specification for :

**/chat Agent (OpenAI Agents SDK + Gemini + Context7)**

---

## HARD RULES (MANDATORY)

* ❌ **No classes** (functional / procedural style only)
* ✅ **Single agent file only**
* ✅ **OpenAI Agents SDK only** (no raw chat completions)
* ✅ **Gemini model only** (loaded from `.env`)
* ✅ **Context7 MCP must ALWAYS be consulted** for:

  * OpenAI Agents SDK documentation
  * Tool schema
  * Runner invocation
* ❌ No persistence (no DB, no files, no chat history)
* ❌ No direct Qdrant access (reuse Step-1 retrieval only)

---

## GOAL

Add a `/chat` endpoint that passes a user question to an agent, retrieves context when required, and returns a grounded answer with citations.

---

## ENDPOINT

```
POST /chat
```

### Input (RAG)

```json
{
  "query": "Explain ROS 2 nodes",
  "mode": "rag",
  "top_k": 6,
  "filters": { "doc_id": "MODULE-1/ROS2-ARCHITECTURE.md" }
}
```

### Input (Selection)

```json
{
  "query": "Summarize this",
  "mode": "selection",
  "selected_text": "Highlighted text"
}
```

### Output

```json
{
  "answer": "string",
  "citations": [{ "doc_id": "string", "chunk_id": 1 }],
  "mode": "rag"
}
```

---

## HOW QUESTION REACHES AGENT

* `/chat` receives HTTP request
* Extract `query` (and `selected_text` if present)
* Build agent input string
* Call OpenAI Agents SDK Runner
* Agent processes input and returns structured result

---

## AGENT FILE (ONE FILE ONLY)

```
backend/src/api/chat_agent.py
```

This file must contain:

* Context7 MCP calls (first-class dependency)
* Agent creation (no classes)
* System prompt
* Tool definition
* Runner execution

No additional agent files
No agent folders

---

## AGENT INPUT RULES

* RAG mode
  agent_input = query

* Selection mode
  SELECTED TEXT:
  <selected_text>

  QUESTION: <query>

---

## TOOL (RAG MODE ONLY)

* Tool name: retrieve_chunks
* Defined in the same file
* Calls existing rag.service.retrieve()
* Forbidden in selection mode
* No direct Qdrant access

---

## AGENT RULES

* Answer only from provided context
* If context is insufficient → respond: "Not enough information"
* Always return citations in RAG mode
* Never hallucinate
* Never use external knowledge

---

## ENV (.env)

* GEMINI_API_KEY

---

## DONE WHEN

* /chat works in both RAG and Selection modes
* Gemini model is used
* Context7 MCP is consulted
* No classes, no persistence
* Citations returned correctly
* Tests pass"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ask Questions Using RAG Mode (Priority: P1)

As a user, I want to ask questions about robotics and humanoid content so that I can get accurate, contextually-relevant answers with citations to the source material. The system should retrieve relevant chunks from the knowledge base and ground the response in that information.

**Why this priority**: This is the core functionality of the RAG system and provides the primary value proposition of contextual, sourced answers.

**Independent Test**: Can be fully tested by sending a query in RAG mode and verifying that the response contains grounded answers with proper citations to source documents.

**Acceptance Scenarios**:

1. **Given** user has a question about robotics content, **When** user sends a POST request to /chat with mode "rag" and a query, **Then** system returns a response with an answer grounded in retrieved context and citation information
2. **Given** user has a question with specific document filters, **When** user sends a POST request to /chat with mode "rag", query, and filters, **Then** system returns a response with answers based only on the filtered documents

---

### User Story 2 - Summarize Selected Text (Priority: P2)

As a user, I want to submit selected text with a query so that I can get a summary or analysis of the specific content I've highlighted, without retrieving additional context from the knowledge base.

**Why this priority**: This provides an alternative interaction mode that allows users to work with their own selected content rather than relying on RAG retrieval.

**Independent Test**: Can be fully tested by sending a query in selection mode with selected text and verifying that the agent responds appropriately to the provided content.

**Acceptance Scenarios**:

1. **Given** user has selected text and wants to analyze it, **When** user sends a POST request to /chat with mode "selection", selected_text, and a query, **Then** system returns a response based only on the provided selected text

---

### User Story 3 - Handle Insufficient Information (Priority: P3)

As a user, I want to receive clear feedback when the system cannot answer my question due to insufficient context, so I know when the system cannot provide a reliable answer.

**Why this priority**: This prevents hallucinations and maintains trust in the system by being transparent about limitations.

**Independent Test**: Can be fully tested by sending queries that cannot be answered with available context and verifying that the system returns "Not enough information".

**Acceptance Scenarios**:

1. **Given** user asks a question with insufficient context in either mode, **When** user sends a POST request to /chat, **Then** system returns "Not enough information" response

---

### Edge Cases

- What happens when the query is malformed or invalid?
- How does the system handle extremely long queries or selected text?
- What happens when the Context7 MCP is unavailable or returns an error?
- How does the system handle when the Gemini model is unavailable or returns an error?
- What happens when the retrieval tool fails during RAG mode?
- How does the system handle concurrent requests?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a POST /chat endpoint that accepts user queries in JSON format
- **FR-002**: System MUST support two modes: "rag" for retrieval-augmented generation and "selection" for working with provided text
- **FR-003**: System MUST use the OpenAI Agents SDK to process user queries (no raw chat completions)
- **FR-004**: System MUST use the Gemini model loaded from environment variables for agent processing
- **FR-005**: System MUST consult the Context7 MCP for OpenAI Agents SDK documentation, tool schema, and runner invocation
- **FR-006**: System MUST define and use a retrieve_chunks tool that calls existing rag.service.retrieve() in RAG mode only
- **FR-007**: System MUST NOT allow the retrieve_chunks tool to be used in selection mode
- **FR-008**: System MUST build appropriate agent input based on mode: query-only for RAG mode, selected text with question for selection mode
- **FR-009**: System MUST return responses with "answer", "citations", and "mode" fields
- **FR-010**: System MUST include citation information (doc_id and chunk_id) in RAG mode responses
- **FR-011**: System MUST return "Not enough information" when context is insufficient to answer the query
- **FR-012**: System MUST implement the agent in a single file at backend/src/api/chat_agent.py using functional/procedural style only (no classes)
- **FR-013**: System MUST NOT persist any data (no DB, no files, no chat history)
- **FR-014**: System MUST NOT access Qdrant directly (reuse Step-1 retrieval only)
- **FR-015**: System MUST answer only from provided context and never hallucinate
- **FR-016**: System MUST validate input parameters and return appropriate error responses for invalid inputs

### Key Entities *(include if feature involves data)*

- **Query Request**: Contains user's question, mode (rag/selection), optional parameters like top_k and filters
- **Query Response**: Contains the answer, citations array with doc_id and chunk_id, and the mode
- **Selected Text**: Text provided by user in selection mode for analysis
- **Citation**: Reference to source document containing doc_id and chunk_id

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can submit queries in both RAG and selection modes and receive appropriate responses within 10 seconds
- **SC-002**: The system successfully processes 95% of valid queries without errors
- **SC-003**: Responses in RAG mode include proper citation information with document IDs and chunk IDs
- **SC-004**: The system returns "Not enough information" for queries that cannot be answered with available context, preventing hallucinations
- **SC-005**: The system correctly restricts the retrieve_chunks tool to RAG mode only and prevents its use in selection mode
- **SC-006**: All agent functionality is implemented in a single file (backend/src/api/chat_agent.py) without classes
- **SC-007**: The Context7 MCP is successfully consulted for OpenAI Agents SDK documentation, tool schema, and runner invocation
