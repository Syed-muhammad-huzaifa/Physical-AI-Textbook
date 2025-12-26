# Quickstart Guide: Chat Agent with OpenAI SDK and Gemini

## Prerequisites

- Python 3.11+
- FastAPI
- OpenAI Agents SDK
- Google Generative AI (for Gemini model)
- Context7 MCP server
- Existing RAG service (rag.service)

## Environment Setup

1. Set up environment variables:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

2. Ensure Context7 MCP server is running and accessible

3. Verify RAG service is properly configured with Qdrant connection

## Implementation Steps

### 1. Create the Agent File

Create `backend/src/api/chat_agent.py` with:

- Context7 MCP consultation for OpenAI Agents SDK usage
- System instructions for grounded responses
- retrieve_chunks tool definition (RAG mode only)
- Agent creation using Gemini model
- Mode-specific input handling
- Response parsing with citation extraction

### 2. Update Router

Register the `/chat` endpoint in `backend/src/api/router.py`:

```python
from .chat_agent import router as chat_router

api_router.include_router(chat_router, prefix="/chat")
```

### 3. Testing

Run tests to verify:
- RAG mode: retrieval tool is called, citations returned
- Selection mode: retrieval tool is not called
- No class definitions in chat_agent.py
- Response schema compliance
- Error handling

## Usage Examples

### RAG Mode Request
```json
{
  "query": "Explain ROS 2 nodes",
  "mode": "rag",
  "top_k": 6,
  "filters": { "doc_id": "MODULE-1/ROS2-ARCHITECTURE.md" }
}
```

### Selection Mode Request
```json
{
  "query": "Summarize this",
  "mode": "selection",
  "selected_text": "Complex robotic systems require careful coordination..."
}
```

## Expected Response
```json
{
  "answer": "string",
  "citations": [{ "doc_id": "string", "chunk_id": 1 }],
  "mode": "rag"
}
```

## Key Constraints

- No classes (functional/procedural style only)
- Single file implementation
- No persistence (no DB/files/chat history)
- Gemini model usage
- Context7 MCP consultation
- Tool access restricted by mode