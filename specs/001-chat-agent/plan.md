# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a /chat endpoint that passes user questions to an OpenAI Agent using Gemini model. The agent will operate in two modes: RAG mode (retrieving context from knowledge base) and selection mode (working with provided selected text only). The system must return grounded answers with citations and follow all constitutional requirements for deterministic AI responses. The implementation will be in a single file (backend/src/api/chat_agent.py) using functional style, with the Context7 MCP consulted for OpenAI Agents SDK usage patterns.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: FastAPI, OpenAI Agents SDK, Google Generative AI (Gemini), Context7 MCP
**Storage**: N/A (stateless, no persistence required)
**Testing**: pytest
**Target Platform**: Linux server
**Project Type**: Web backend service
**Performance Goals**: <10 seconds response time for queries, 95% success rate
**Constraints**: No classes allowed (functional/procedural style only), no persistence (no DB/files/chat history), single file implementation only
**Scale/Scope**: Single endpoint supporting both RAG and selection modes

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Grounded & Deterministic AI Compliance
- ✅ All AI responses will be grounded in retrieved content from existing RAG system
- ✅ System will respond "Not enough information" when context is insufficient
- ✅ No hallucinated or fabricated answers will be generated

### RAG Chatbot Rules Compliance
- ✅ Answers will come ONLY from retrieved book content via existing rag.service.retrieve()
- ✅ System will say it does not know when no relevant content exists
- ✅ Source chapter/section will be referenced via citations (doc_id, chunk_id)
- ✅ Selected-text-only answering will be supported (no global context in selection mode)
- ✅ Selected-text mode will NOT use global context (retrieval tool disabled)
- ✅ External internet access will be forbidden (using existing rag.service only)

### Platform & Stack Standards Compliance
- ✅ Backend framework: FastAPI (already established in project)
- ✅ RAG orchestration: OpenAI Agents SDK (as required by spec)
- ✅ Vector DB: Using existing Qdrant Cloud via rag.service.retrieve() (no direct access)

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── api/
│   │   ├── chat_agent.py      # New single agent file (required)
│   │   └── router.py          # Updated to register /chat endpoint
│   ├── models/
│   ├── services/
│   │   └── rag.service.py     # Existing service (for rag.service.retrieve)
│   └── config/
└── tests/
    └── api/
        └── test_chat_agent.py # New tests for the chat endpoint
```

**Structure Decision**: Web application with backend service following existing project structure. The new chat agent functionality will be implemented in a single file backend/src/api/chat_agent.py as required by the specification, with the endpoint registered in the existing router.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
