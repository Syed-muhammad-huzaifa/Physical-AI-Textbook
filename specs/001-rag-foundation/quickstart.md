# Quickstart: RAG Foundation

## Prerequisites

1. **Environment Variables**: Ensure the following are set in your `.env` file:
   ```
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   FASTEMBED_MODEL=your_fastembed_model (e.g., "all-MiniLM-L6-v2")
   QDRANT_COLLECTION=rag_chunks
   CHUNK_SIZE=800
   CHUNK_OVERLAP=120
   DOCS_DIR=book-source/docs
   ```

2. **Dependencies**: Install required packages:
   ```bash
   cd backend
   poetry add fastembed qdrant-client
   poetry add pytest --group dev  # for testing
   ```

3. **Directory Structure**: Ensure `book-source/docs/` directory exists with markdown files

## Setup Steps

1. **Create the rag module structure**:
   ```bash
   mkdir -p backend/src/rag
   ```

2. **Add rag endpoints to the API router** in `backend/src/api/router.py`:
   ```python
   from fastapi import APIRouter
   from .auth import router as auth_router
   from .rag import router as rag_router  # Add this import

   # Main API router that includes all sub-routers
   api_router = APIRouter()

   # Include authentication routes
   api_router.include_router(auth_router)

   # Include RAG routes (add this line)
   api_router.include_router(rag_router, prefix="/rag")
   ```

3. **Start the service**:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

## Basic Usage

### 1. Health Check
Verify the system is running:
```bash
curl http://localhost:8000/api/v1/health
```

### 2. Ingest Documents
Process all markdown files from the configured directory:
```bash
curl -X POST http://localhost:8000/api/v1/rag/ingest-dir \
  -H "Content-Type: application/json" \
  -d '{"wipe_collection": false}'
```

### 3. Retrieve Information
Search for relevant information:
```bash
curl -X POST http://localhost:8000/api/v1/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is a ROS 2 node?",
    "top_k": 6
  }'
```

## Testing

### Unit Tests
Run unit tests for the RAG components:
```bash
cd backend
pytest tests/unit/test_chunking.py
pytest tests/unit/test_loader.py
```

### Integration Test
Run integration test for the full RAG workflow:
```bash
cd backend
pytest tests/integration/test_rag_integration.py
```

## Configuration

The system uses environment variables for configuration:

- `QDRANT_URL`: URL for Qdrant vector database
- `QDRANT_API_KEY`: API key for Qdrant (if required)
- `FASTEMBED_MODEL`: Model to use for embeddings (default: "all-MiniLM-L6-v2")
- `QDRANT_COLLECTION`: Name of the Qdrant collection (default: "rag_chunks")
- `CHUNK_SIZE`: Size of text chunks (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 120)
- `DOCS_DIR`: Directory containing markdown files (default: "book-source/docs")