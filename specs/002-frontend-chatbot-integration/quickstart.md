# Quickstart Guide: Frontend Chatbot Integration for Docusaurus

## Prerequisites

- Node.js 18+ with npm/yarn
- Docusaurus v3.x project
- Existing backend with `/chat` endpoint (from previous implementation)
- TypeScript (for type safety)

## Environment Setup

1. Ensure backend `/chat` endpoint is running and accessible
2. Verify CORS configuration allows requests from your Docusaurus site
3. Make sure the backend environment variables are properly configured

## Implementation Steps

### 1. Create Chatbot Components

Create the following directory structure and components:

```
src/components/ChatBot/
├── ChatBot.tsx
├── ChatWindow.tsx
├── MessageBubble.tsx
├── TextInput.tsx
├── SelectionHandler.tsx
└── api.ts
```

### 2. Update Docusaurus Configuration

Add the chatbot components to your `docusaurus.config.js`:

```javascript
// Add to plugins or themes as needed
module.exports = {
  // ... existing config
  themes: [
    // ... existing themes
  ],
  plugins: [
    // ... existing plugins
  ],
};
```

### 3. Wrap App with Chat Context

Update `src/theme/Root.tsx` to wrap the app with chatbot context:

```tsx
// Import and use the chatbot context provider
```

### 4. Add Floating Button

Add a floating chat button that appears on all pages.

### 5. Configure API Communication

Set up the API communication layer to connect with the backend `/chat` endpoint.

## Usage Examples

### RAG Mode Query
```json
{
  "query": "Explain ROS 2 nodes",
  "mode": "rag",
  "top_k": 6,
  "filters": { "doc_id": "MODULE-1/ROS2-ARCHITECTURE.md" }
}
```

### Selection Mode Query
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

## Key Integration Points

- **Text Selection Detection**: Uses browser Selection API to detect user selections
- **Persistent Interface**: Floating chat window that stays accessible across page navigations
- **Conversation History**: Stores conversation in browser localStorage
- **Responsive Design**: Works on mobile, tablet, and desktop devices
- **Accessibility**: Proper ARIA labels and keyboard navigation support
- **Error Handling**: Graceful handling of API errors and network issues