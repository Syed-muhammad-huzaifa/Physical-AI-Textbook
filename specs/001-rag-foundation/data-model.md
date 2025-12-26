# Data Model: RAG Foundation

## Core Entities

### Document Chunk
**Description**: A segment of markdown text with associated metadata and embedding vector

**Fields**:
- `doc_id` (string): Relative path from book-source/docs/ (e.g., "MODULE-1/ROS2-ARCHITECTURE.md")
- `chunk_id` (int): Sequential ID for chunks within a document
- `source` (string): Same as doc_id, for compatibility
- `text` (string): The actual chunked text content
- `embedding` (list[float]): Vector representation of the text content

**Relationships**: Belongs to a Document

### Document
**Description**: A markdown file identified by its relative path with associated chunks

**Fields**:
- `doc_id` (string): Relative path from book-source/docs/ (e.g., "MODULE-1/ROS2-ARCHITECTURE.md")
- `source` (string): Same as doc_id
- `chunks` (list[DocumentChunk]): Associated text chunks

### Query
**Description**: A text input that is embedded and used to find similar chunks in the vector database

**Fields**:
- `text` (string): The query text
- `vector` (list[float]): Embedded representation of the query text
- `top_k` (int): Number of results to return (default: 6)
- `filters` (dict): Optional filters (e.g., {"doc_id": "MODULE-1/ROS2-ARCHITECTURE.md"})

### Qdrant Collection
**Description**: A container for storing document chunks with their embeddings and metadata

**Fields**:
- `name` (string): From QDRANT_COLLECTION environment variable (default: "rag_chunks")
- `vector_size` (int): Dimension of the embedding vectors
- `chunks` (list[DocumentChunk]): All document chunks stored in this collection

## Payload Structure (for Qdrant storage)

When storing chunks in Qdrant, the payload will contain:
```json
{
  "doc_id": "string",
  "chunk_id": "int",
  "source": "string",
  "text": "string"
}
```

## Retrieval Result Structure

When retrieving chunks from Qdrant, the result will contain:
```json
{
  "score": "float",
  "doc_id": "string",
  "chunk_id": "int",
  "source": "string",
  "text": "string"
}
```

## Validation Rules

1. **Document ID Validation**:
   - Must be a valid relative path from book-source/docs/
   - Must end with ".md" extension
   - Must not contain path traversal sequences (../)

2. **Chunk Validation**:
   - `chunk_id` must be a positive integer
   - `text` must not be empty
   - `text` length should respect CHUNK_SIZE configuration

3. **Query Validation**:
   - Query text must not be empty
   - top_k must be between 1 and 100
   - Filters must be valid field names

## State Transitions

1. **Document Ingestion Flow**:
   - Document identified → Content read → Chunked → Embedded → Stored in Qdrant
   - On re-ingestion: Existing chunks are upserted (overwritten) based on doc_id + chunk_id

2. **Query Retrieval Flow**:
   - Query received → Embedded → Qdrant search → Results returned with metadata