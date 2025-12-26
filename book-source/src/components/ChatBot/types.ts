// TypeScript type definitions for chat messages and state

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isLoading?: boolean;
}

export interface Citation {
  doc_id: string;
  chunk_id: number;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  mode: 'rag' | 'selection';
}

export interface ChatRequest {
  query: string;
  mode: 'rag' | 'selection';
  selected_text?: string;
  top_k?: number;
  filters?: Record<string, any>;
}

export interface ChatSession {
  id: string;
  messages: ChatMessage[];
  currentMode: 'rag' | 'selection';
  selectedText?: string;
  isActive: boolean;
}

export interface TextSelection {
  text: string;
  elementId?: string;
  pageUrl: string;
  timestamp: Date;
}

export interface ChatState {
  session: ChatSession;
  isLoading: boolean;
  error?: string;
  isVisible: boolean;
  isMinimized: boolean;
}

export interface StoredConversation {
  sessionId: string;
  messages: (Omit<ChatMessage, 'timestamp'> & { timestamp: string }); // Serialized for localStorage
  lastAccessed: string; // ISO string for localStorage
  pageContext: string;
}