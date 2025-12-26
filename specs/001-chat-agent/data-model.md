# Data Model: Chat Agent with OpenAI SDK and Gemini

## Request Models

### ChatRequest
- **query**: string - The user's question or query
- **mode**: string - Either "rag" or "selection" to determine processing mode
- **top_k**: integer (optional) - Number of chunks to retrieve (default: 6)
- **filters**: object (optional) - Filters for document retrieval (e.g., {"doc_id": "filename.md"})
- **selected_text**: string (optional) - Text provided in selection mode

### Validation Rules
- query: Required, non-empty string
- mode: Required, must be either "rag" or "selection"
- top_k: If provided, must be between 1 and 100
- filters: If provided, must be a valid filter object
- selected_text: Required when mode is "selection"

## Response Models

### ChatResponse
- **answer**: string - The agent's response to the query
- **citations**: array of Citation objects - References to source documents
- **mode**: string - The processing mode ("rag" or "selection")

### Citation
- **doc_id**: string - Document identifier
- **chunk_id**: integer - Chunk identifier within the document

### Validation Rules
- answer: Required, non-empty string
- citations: Required, array of Citation objects
- mode: Required, must match the input mode

## Internal Models

### AgentInput
- **content**: string - Formatted input for the agent based on mode
  - RAG mode: Direct query string
  - Selection mode: "SELECTED TEXT:\n<selected_text>\n\nQUESTION:\n<query>"

### AgentOutput
- **content**: string - Raw response from the agent
- **tool_calls**: array - Tool calls made during processing (RAG mode only)

## State Transitions

The system is stateless with no persistence, so there are no state transitions. Each request is processed independently.

## Relationships

- ChatRequest contains a mode that determines whether to use retrieval tools
- ChatResponse citations are derived from the retrieved chunks when in RAG mode
- Selection mode responses do not contain citations as they don't use retrieval