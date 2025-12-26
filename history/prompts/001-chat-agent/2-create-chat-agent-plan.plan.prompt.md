---
id: 2
title: Create-chat-agent-plan
stage: plan
date: 2025-12-20
surface: agent
model: claude-sonnet-4-5-20250929
feature: chat-agent
branch: 001-chat-agent
user: syedhuzaifa
command: "/sp.plan"
labels: [planning, chat-agent, openai-agents, gemini, rag]
links:
  spec: specs/001-chat-agent/spec.md
  plan: specs/001-chat-agent/plan.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-chat-agent/plan.md
  - specs/001-chat-agent/research.md
  - specs/001-chat-agent/data-model.md
  - specs/001-chat-agent/quickstart.md
  - specs/001-chat-agent/contracts/chat-api.yaml
tests:
  - No tests created yet
---

## Prompt

Here is the **concise, execution-ready PLAN** for the Step-2 specification you locked.
No extra theory — just **what to do, in order**.

---

# STEP-2 IMPLEMENTATION PLAN

**/chat Agent (OpenAI Agents SDK + Gemini + Context7)**

---

## 1️⃣ Create the single agent file

**Add only one new file:**

```
backend/src/api/chat_agent.py
```

❌ Do NOT create any agent folders
❌ Do NOT create classes

---

## 2️⃣ Consult Context7 MCP (first thing in the file)

* Call **Context7 MCP server**
* Fetch:

  * Latest **OpenAI Agents SDK** usage
  * Tool decorator syntax
  * Runner invocation schema
* Use **exact schema** returned by Context7
* Do not guess or hardcode SDK APIs

---

## 3️⃣ Define system instructions (inline)

* Add a strict system prompt:

  * Answer only from provided context
  * No external knowledge
  * No hallucinations
  * Return citations (`doc_id`, `chunk_id`)
  * Say "Not enough information" when context is missing

---

## 4️⃣ Define retrieval tool (same file)

* Define `retrieve_chunks` using Agents SDK tool mechanism
* Tool must:

  * Call `rag.service.retrieve(...)`
  * Return `{ doc_id, chunk_id, text }`
* Enable tool **only in RAG mode**
* Ensure tool **cannot be called** in selection mode

---

## 5️⃣ Build agent (functional style only)

* Create agent as a **plain Agent object**
* Use:

  * Gemini model from `.env`
  * System instructions
  * Tool list (RAG mode only)
* ❌ No classes
* ❌ No inheritance

---

## 6️⃣ Construct agent input

Inside `/chat` handler logic:

### RAG mode

```
agent_input = query
```

### Selection mode

```
SELECTED TEXT:
<selected_text>

QUESTION:
<query>
```

---

## 7️⃣ Run agent via Agents SDK

* Invoke agent using **Runner**
* Pass `agent_input` as a string
* Let SDK handle reasoning + tool calls

---

## 8️⃣ Parse agent output

* Extract:

  * `answer`
  * `citations`
* Enforce response schema:

```json
{
  "answer": "string",
  "citations": [],
  "mode": "rag | selection"
}
```

---

## 9️⃣ Expose `/chat` endpoint

* Register `/chat` route in:

```
backend/src/api/router.py
```

* Ensure existing routes remain untouched

---

## 🔟 Tests (must pass)

### Unit

* Selection mode → retrieval tool **never called**
* No class definitions in `chat_agent.py`

### Integration

* RAG mode:

  * Retrieval tool called
  * Answer returned
  * Valid citations returned
* No files / DB created (stateless)

---

## ✅ DONE WHEN

* `/chat` endpoint works
* Gemini model used
* Context7 MCP consulted
* No classes
* No persistence
* All tests pass

## Response snapshot

Created a comprehensive implementation plan for the chat agent feature. The plan includes:

1. Updated the implementation plan file (plan.md) with:
   - Technical context including Python 3.11, FastAPI, OpenAI Agents SDK, Google Generative AI
   - Constitution check confirming compliance with project standards
   - Project structure showing the new chat_agent.py file and router updates
   - Summary of the implementation approach

2. Created research.md with decisions about:
   - OpenAI Agents SDK implementation
   - Context7 MCP consultation
   - Gemini model configuration
   - Single file implementation
   - rag.service.retrieve() integration
   - Mode-specific tool access

3. Created data-model.md defining:
   - Request/Response models (ChatRequest, ChatResponse, Citation)
   - Validation rules
   - Internal models (AgentInput, AgentOutput)
   - State transitions (none, as system is stateless)

4. Created quickstart.md with:
   - Prerequisites and environment setup
   - Implementation steps
   - Usage examples
   - Key constraints

5. Created API contract (contracts/chat-api.yaml) defining:
   - OpenAPI specification for the /chat endpoint
   - Request/response schemas
   - Error handling

The plan is complete and ready for the implementation phase. All Phase 0 (research) and Phase 1 (design & contracts) requirements have been fulfilled.

## Outcome

- ✅ Impact: Complete implementation plan created with research, data models, contracts, and quickstart guide
- 🧪 Tests: No tests created yet (will be created in implementation phase)
- 📁 Files: Created plan.md, research.md, data-model.md, quickstart.md, and contracts/chat-api.yaml
- 🔁 Next prompts: Ready for task generation with /sp.tasks
- 🧠 Reflection: Plan addresses all requirements from the specification with proper research and design artifacts

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Proceed to task generation phase