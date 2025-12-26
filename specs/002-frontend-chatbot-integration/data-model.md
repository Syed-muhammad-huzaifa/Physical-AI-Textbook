# Data Model: Frontend Chatbot Integration for Docusaurus

## Request Models

### Frontend Chat Request
- **query**: string - The user's question or input text
- **mode**: string - Either "rag" for full book context or "selection" for selected text only
- **selected_text**: string (optional) - Text selected by user in selection mode
- **top_k**: number (optional) - Number of chunks to retrieve (default: 6)
- **filters**: object (optional) - Document filters for RAG mode (e.g., {"doc_id": "filename.md"})

### Validation Rules
- query: Required, non-empty string
- mode: Required, must be either "rag" or "selection"
- selected_text: Required when mode is "selection"
- top_k: If provided, must be between 1 and 100
- filters: If provided, must be a valid filter object

## Response Models

### Frontend Chat Response
- **answer**: string - The chatbot's response to the query
- **citations**: array of Citation objects - References to source documents
- **mode**: string - The processing mode ("rag" or "selection")

### Citation
- **doc_id**: string - Document identifier
- **chunk_id**: number - Chunk identifier within the document

### Validation Rules
- answer: Required, non-empty string
- citations: Required, array of Citation objects
- mode: Required, must match the input mode

## Internal Models

### ChatMessage
- **id**: string - Unique identifier for the message
- **role**: string - Either "user" or "assistant"
- **content**: string - The message content
- **timestamp**: Date - When the message was created
- **isLoading**: boolean - Whether the message is still being processed

### ChatSession
- **id**: string - Unique session identifier
- **messages**: array of ChatMessage - Conversation history
- **currentMode**: string - Current interaction mode ("rag" or "selection")
- **selectedText**: string (optional) - Currently selected text
- **isActive**: boolean - Whether the chat is currently open

### TextSelection
- **text**: string - The selected text content
- **elementId**: string (optional) - DOM element ID where selection occurred
- **pageUrl**: string - URL of the page where selection occurred
- **timestamp**: Date - When the selection was made

## State Management

### ChatState
- **session**: ChatSession object - Current conversation state
- **isLoading**: boolean - Whether an API request is in progress
- **error**: string (optional) - Error message if API request failed
- **isVisible**: boolean - Whether the chat interface is visible
- **isMinimized**: boolean - Whether the chat window is minimized

## Relationships

- ChatSession contains multiple ChatMessage objects forming the conversation history
- ChatMessage can have multiple Citation objects when in RAG mode
- TextSelection is associated with a ChatSession when in selection mode
- ChatState manages the overall UI state of the chat interface

## Storage Model

### StoredConversation (for localStorage)
- **sessionId**: string - Unique identifier for the conversation
- **messages**: array of serialized ChatMessage - Conversation history
- **lastAccessed**: Date - When the conversation was last accessed
- **pageContext**: string - The page where the conversation was initiated