// Browser storage utilities for conversation history persistence
import { ChatSession, StoredConversation } from './types';

const STORAGE_KEY = 'docusaurus-chat-session';
const HISTORY_LIMIT = 50; // Limit conversation history to prevent storage overflow

/**
 * Save conversation to localStorage
 */
export const saveConversation = (session: ChatSession): void => {
  try {
    const serialized: StoredConversation = {
      sessionId: session.id,
      messages: session.messages.map(message => ({
        ...message,
        timestamp: message.timestamp.toISOString()
      })),
      lastAccessed: new Date().toISOString(),
      pageContext: window.location.pathname,
    };

    localStorage.setItem(STORAGE_KEY, JSON.stringify(serialized));
  } catch (error) {
    console.error('Failed to save conversation to localStorage:', error);
  }
};

/**
 * Load conversation from localStorage
 */
export const loadConversation = (): ChatSession | null => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return null;

    const parsed: StoredConversation = JSON.parse(stored);

    // Deserialize timestamps
    const deserialized: ChatSession = {
      ...parsed,
      messages: parsed.messages.map(message => ({
        ...message,
        timestamp: new Date(message.timestamp),
      })),
    };

    return deserialized;
  } catch (error) {
    console.error('Failed to load conversation from localStorage:', error);
    return null;
  }
};

/**
 * Clear conversation from localStorage
 */
export const clearConversation = (): void => {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.error('Failed to clear conversation from localStorage:', error);
  }
};

/**
 * Get storage usage information
 */
export const getStorageInfo = (): { used: number; limit: number; percentage: number } => {
  try {
    const used = JSON.stringify(localStorage).length;
    const limit = 5 * 1024 * 1024; // 5MB typical limit
    const percentage = (used / limit) * 100;

    return {
      used,
      limit,
      percentage
    };
  } catch (error) {
    console.error('Failed to get storage info:', error);
    return { used: 0, limit: 0, percentage: 0 };
  }
};

/**
 * Clean up old conversations (if we extend to support multiple sessions)
 */
export const cleanupOldConversations = (): void => {
  // For now, we only store one conversation, but this could be extended
  // to manage multiple conversations by date or other criteria
  try {
    const conversation = loadConversation();
    if (conversation) {
      // Keep only the most recent messages if we exceed the limit
      if (conversation.messages.length > HISTORY_LIMIT) {
        const trimmedMessages = conversation.messages.slice(-HISTORY_LIMIT);
        const trimmedSession = {
          ...conversation,
          messages: trimmedMessages
        };
        saveConversation(trimmedSession);
      }
    }
  } catch (error) {
    console.error('Failed to cleanup conversations:', error);
  }
};