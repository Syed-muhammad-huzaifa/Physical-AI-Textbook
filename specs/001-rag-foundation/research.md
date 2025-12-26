# Research: RAG Foundation Implementation

## Decision: FastEmbed as embedding model
**Rationale**: FastEmbed is a lightweight, efficient library for text embeddings that supports multiple models including all-MiniLM-L6-v2, which is suitable for document retrieval. It's faster and more resource-efficient than OpenAI embeddings for local deployment and meets the requirement of using a single embedding model consistently.

**Alternatives considered**:
- SentenceTransformers: More complex but more model options
- OpenAI embeddings: Requires API calls and costs money
- Instructor embeddings: More complex setup for this use case

## Decision: Qdrant as vector database
**Rationale**: Qdrant is chosen as it's specified in the constitution as the required vector database. It supports efficient similarity search, has good Python client support, and can handle the expected scale of document chunks.

**Alternatives considered**:
- Pinecone: Cloud-only, vendor lock-in concerns
- Weaviate: Good alternative but constitution specifies Qdrant
- FAISS: Facebook's library but requires more manual management

## Decision: Character-based chunking approach
**Rationale**: Character-based chunking with configurable size (800) and overlap (120) provides consistent chunking regardless of document structure. This approach is simple to implement and meets the requirements specified.

**Alternatives considered**:
- Sentence-based chunking: More semantically aware but potentially inconsistent sizes
- Recursive chunking: More complex but could preserve document structure better
- Markdown-aware chunking: Could respect headers but adds complexity

## Decision: Environment-based configuration
**Rationale**: Using environment variables for configuration (QDRANT_URL, QDRANT_API_KEY, FASTEMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR) provides flexibility for different environments while meeting the requirement of not hardcoding values.

**Alternatives considered**:
- Configuration files: Would require additional file management
- Database configuration: Overkill for simple settings
- Command-line parameters: Less flexible for deployment

## Decision: Stable ID generation using doc_id + chunk_id
**Rationale**: Creating stable point IDs based on doc_id + chunk_id ensures that re-ingestion correctly overwrites existing documents as required by the specification, while maintaining consistency across restarts.

**Alternatives considered**:
- Random UUIDs: Would not support proper upsert behavior
- Sequential integers: Would not maintain document-chunk relationship
- Hash-based: More complex but similar result

## Decision: FastAPI dependency integration
**Rationale**: Adding FastEmbed and Qdrant client as dependencies to the existing pyproject.toml will extend the current backend without requiring a new project structure, maintaining consistency with the existing architecture.

**Required dependencies to add**:
- fastembed>=0.3.1
- qdrant-client>=1.9.0
- pytest>=8.0.0 (for testing)

## Decision: Integration with existing auth system
**Rationale**: The RAG endpoints should integrate with the existing better-auth system to maintain security consistency across the application. The health check endpoint may be public, but ingestion and retrieval endpoints should require authentication.

**Implementation approach**: Use existing auth dependencies from the current backend structure.