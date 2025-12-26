# Research Summary: Chat Agent with OpenAI SDK and Gemini

## Decision: OpenAI Agents SDK Implementation
**Rationale**: The specification requires using OpenAI Agents SDK to process user queries. This provides a structured way to create agents with tools and system instructions, which is perfect for our RAG use case.

## Decision: Context7 MCP Consultation
**Rationale**: The specification mandates consulting Context7 MCP for OpenAI Agents SDK documentation, tool schema, and runner invocation. This ensures we use the correct and up-to-date APIs rather than relying on potentially outdated documentation.

## Decision: Gemini Model Configuration
**Rationale**: The specification requires using the Gemini model loaded from environment variables. This provides consistency with the project's AI strategy and allows for easy configuration across environments.

## Decision: Single File Implementation (chat_agent.py)
**Rationale**: The specification mandates implementing the agent in a single file at backend/src/api/chat_agent.py using functional/procedural style only. This keeps the implementation contained and follows the project's architectural constraints.

## Decision: rag.service.retrieve() Integration
**Rationale**: The existing rag.service module provides the retrieve() function that we need for the RAG mode. Using this existing service ensures consistency with the current RAG implementation and avoids duplicating retrieval logic.

## Decision: Mode-Specific Tool Access
**Rationale**: The retrieve_chunks tool must only be available in RAG mode, not in selection mode. This is implemented by conditionally providing the tool based on the mode specified in the request.

## Decision: Input Format Handling
**Rationale**: The agent input must be formatted differently based on mode:
- RAG mode: Direct query string
- Selection mode: "SELECTED TEXT:\n<selected_text>\n\nQUESTION:\n<query>"

This ensures the agent receives the appropriate context for each mode.

## Decision: Response Schema Compliance
**Rationale**: The response must include "answer", "citations", and "mode" fields to match the specification. The citations field must contain doc_id and chunk_id information when available in RAG mode.

## Alternatives Considered:

1. **Raw Chat Completions vs OpenAI Agents SDK**: Raw chat completions were rejected as the specification explicitly requires OpenAI Agents SDK.

2. **Multiple Files vs Single File**: Multiple files were rejected as the specification explicitly requires a single file implementation.

3. **Direct Qdrant Access vs rag.service**: Direct Qdrant access was rejected as the specification forbids it and requires reusing Step-1 retrieval only.

4. **Class-based vs Functional Style**: Class-based implementation was rejected as the specification explicitly requires functional/procedural style only.

## Key Findings:

1. The rag.service.retrieve() method returns a list of dictionaries with keys: score, doc_id, chunk_id, source, text
2. The existing RAG system uses Qdrant for vector storage and FastEmbed for embeddings
3. The system already has authentication implemented via better-auth
4. The router system allows including sub-routers with prefixes
5. The existing error handling patterns should be followed for consistency