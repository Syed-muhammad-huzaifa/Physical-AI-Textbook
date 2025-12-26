# Humanoid Robotics Textbook Backend

This is the backend service for the Humanoid Robotics Textbook platform, providing APIs for user authentication, onboarding, and RAG (Retrieval Augmented Generation) functionality.

## Features

- User authentication and management
- Onboarding data collection
- RAG (Retrieval Augmented Generation) system for document search and retrieval

## RAG (Retrieval Augmented Generation) System

The RAG system allows users to search through the robotics textbook content by:

1. **Ingestion**: Processing markdown files from `book-source/docs/`, chunking them, generating embeddings using FastEmbed, and storing in Qdrant vector database
2. **Retrieval**: Searching for relevant content based on user queries
3. **Health Check**: Monitoring system connectivity

### API Endpoints

- `GET /api/v1/rag/health` - Check system health
- `POST /api/v1/rag/ingest-dir` - Ingest markdown files from directory
- `POST /api/v1/rag/retrieve` - Retrieve relevant chunks for a query

### Configuration

The RAG system uses the following environment variables:

- `QDRANT_URL`: URL for Qdrant vector database
- `QDRANT_API_KEY`: API key for Qdrant (if required)
- `QDRANT_COLLECTION`: Name of the Qdrant collection (default: "rag_chunks")
- `FASTEMBED_MODEL`: Model to use for embeddings (default: "all-MiniLM-L6-v2")
- `EMBEDDING_BATCH_SIZE`: Batch size for embedding/upsert to avoid high memory usage (default: 32)
- `CHUNK_SIZE`: Size of text chunks (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 120)
- `DOCS_DIR`: Directory containing markdown files (default: "book-source/docs")

## Setup

1. Install dependencies: `poetry install`
2. Set up environment variables in `.env`
3. Start the service: `poetry run uvicorn main:app --reload`

## Development

This backend is built with FastAPI and follows the project architecture as defined in the project constitution.
