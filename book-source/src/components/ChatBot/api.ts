// API service to communicate with backend /chat endpoint
import { ChatRequest, ChatResponse } from './types';

// Use a default backend URL or get from window environment in browser
// Point to FastAPI backend; include /api/v1 prefix to match router setup
const BACKEND_BASE_URL =
  typeof window !== 'undefined'
    ? (window as any).__docusaurus_chat_backend_url__ || 'http://localhost:8000/api/v1'
    : 'http://localhost:8000/api/v1';

/**
 * Send a chat request to the backend API
 * @param request - The chat request containing query, mode, and optional parameters
 * @returns Promise resolving to the chat response
 */
export const sendChatRequest = async (request: ChatRequest): Promise<ChatResponse> => {
  try {
    const response = await fetch(`${BACKEND_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error sending chat request:', error);
    throw error;
  }
};

/**
 * Ping the backend to check if it's available
 * @returns Promise resolving to true if backend is reachable, false otherwise
 */
export const checkBackendHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${BACKEND_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error('Backend health check failed:', error);
    return false;
  }
};
