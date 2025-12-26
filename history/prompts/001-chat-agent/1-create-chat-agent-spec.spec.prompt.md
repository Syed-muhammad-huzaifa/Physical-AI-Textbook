---
id: 1
title: Create-chat-agent-spec
stage: spec
date: 2025-12-20
surface: agent
model: claude-sonnet-4-5-20250929
feature: chat-agent
branch: 001-chat-agent
user: syedhuzaifa
command: "/sp.specify"
labels: [specification, chat-agent, openai-agents, gemini, rag]
links:
  spec: specs/001-chat-agent/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-chat-agent/spec.md
  - specs/001-chat-agent/checklists/requirements.md
tests:
  - No tests created yet
---

## Prompt

Create a specification for :

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
* Tests pass

## Response snapshot

Created a comprehensive specification for the chat agent feature that includes:

1. User Scenarios & Testing section with three prioritized user stories:
   - P1: Ask Questions Using RAG Mode
   - P2: Summarize Selected Text
   - P3: Handle Insufficient Information

2. Functional Requirements section with 16 specific requirements (FR-001 to FR-016) covering:
   - POST /chat endpoint functionality
   - Support for both RAG and selection modes
   - OpenAI Agents SDK usage with Gemini model
   - Context7 MCP consultation
   - Retrieve_chunks tool implementation
   - Input/output format requirements
   - Data persistence restrictions
   - Hallucination prevention

3. Key Entities section defining Query Request, Query Response, Selected Text, and Citation entities

4. Success Criteria section with 7 measurable outcomes (SC-001 to SC-007) covering:
   - Response time and success rate metrics
   - Proper citation information in responses
   - Hallucination prevention
   - Tool usage restrictions
   - Implementation constraints

Also created a quality checklist at specs/001-chat-agent/checklists/requirements.md to validate the specification completeness.

## Outcome

- ✅ Impact: Created complete feature specification for chat agent with OpenAI Agents SDK and Gemini
- 🧪 Tests: No tests created yet
- 📁 Files: Created spec.md and requirements checklist
- 🔁 Next prompts: Ready for planning phase with /sp.plan
- 🧠 Reflection: Specification addresses all requirements from the original request with clear user stories and measurable success criteria

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Proceed to planning phase