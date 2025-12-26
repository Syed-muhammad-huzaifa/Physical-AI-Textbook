# Implementation Plan: RAG Foundation - Embed Markdown docs → Qdrant → Retrieve

**Branch**: `001-rag-foundation` | **Date**: 2025-12-20 | **Spec**: [link](specs/001-rag-foundation/spec.md)
**Input**: Feature specification from `/specs/001-rag-foundation/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a RAG (Retrieval Augmented Generation) system that reads markdown files from `book-source/docs/`, chunks them, generates embeddings using FastEmbed, stores them in Qdrant vector database, and provides retrieval endpoints. The system will be built as a FastAPI backend with dedicated modules for loading, chunking, embeddings, Qdrant integration, and service orchestration. This implementation will integrate with the existing backend structure and authentication system.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: FastAPI, FastEmbed, Qdrant, Pydantic, python-dotenv, uvicorn
**Storage**: Qdrant vector database (external), existing PostgreSQL for auth (Neon Serverless)
**Testing**: pytest (to be added for unit and integration tests)
**Target Platform**: Linux server (backend service)
**Project Type**: Web backend - extending existing backend structure
**Performance Goals**: 500ms retrieval response time, 99% ingestion success rate, support 1000+ document chunks
**Constraints**: <500ms p95 retrieval, <100MB memory for chunking operations, integration with existing auth system
**Scale/Scope**: Support 1000+ document chunks, handle concurrent retrieval requests, maintain compatibility with existing auth

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- ✅ **Backend framework: FastAPI** - Plan uses FastAPI as required by constitution
- ✅ **Vector DB: Qdrant Cloud** - Plan uses Qdrant as required by constitution
- ✅ **Embedding model consistency** - Plan uses single FastEmbed model as required
- ✅ **Authentication integration** - Plan will integrate with existing better-auth system
- ✅ **Grounded AI responses** - Plan focuses on retrieval (RAG foundation) for grounded responses

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-foundation/
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
│   │   ├── router.py        # Include rag router here
│   │   └── rag.py           # RAG endpoints
│   └── rag/                 # New RAG module
│       ├── loader.py        # Loads ONLY .md files
│       ├── chunking.py      # Text chunking logic
│       ├── embeddings.py    # FastEmbed wrapper
│       ├── qdrant.py        # Qdrant client wrapper
│       └── service.py       # Business logic orchestration
└── tests/
    ├── unit/
    │   ├── test_chunking.py
    │   └── test_loader.py
    ├── integration/
    │   └── test_rag_integration.py
    └── contract/
        └── test_rag_contracts.py
```

**Structure Decision**: Extending existing backend structure - Adding rag module to existing backend to maintain single deployment unit as required by constitution. The new RAG functionality will be integrated into the existing FastAPI application with proper routing and will reuse the existing authentication and database infrastructure.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |
