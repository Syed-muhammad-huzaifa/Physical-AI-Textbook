// React Context for managing chat state
import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { ChatState, ChatSession, ChatMessage } from './types';

// Define action types for the reducer
type ChatAction =
  | { type: 'SET_VISIBLE'; isVisible: boolean }
  | { type: 'SET_MINIMIZED'; isMinimized: boolean }
  | { type: 'SET_LOADING'; isLoading: boolean }
  | { type: 'SET_ERROR'; error: string | undefined }
  | { type: 'ADD_MESSAGE'; message: ChatMessage }
  | { type: 'SET_CURRENT_MODE'; mode: 'rag' | 'selection' }
  | { type: 'SET_SELECTED_TEXT'; selectedText: string | undefined }
  | { type: 'CLEAR_MESSAGES' }
  | { type: 'RESTORE_SESSION'; session: ChatSession }
  | { type: 'START_NEW_SESSION' };

// Initial state
const initialState: ChatState = {
  session: {
    id: `session_${Date.now()}`,
    messages: [],
    currentMode: 'rag',
    isActive: false,
  },
  isLoading: false,
  error: undefined,
  isVisible: false,
  isMinimized: true,
};

// Reducer function
const chatReducer = (state: ChatState, action: ChatAction): ChatState => {
  switch (action.type) {
    case 'SET_VISIBLE':
      return { ...state, isVisible: action.isVisible };
    case 'SET_MINIMIZED':
      return { ...state, isMinimized: action.isMinimized };
    case 'SET_LOADING':
      return { ...state, isLoading: action.isLoading };
    case 'SET_ERROR':
      return { ...state, error: action.error };
    case 'ADD_MESSAGE':
      return {
        ...state,
        session: {
          ...state.session,
          messages: [...state.session.messages, action.message],
        },
      };
    case 'SET_CURRENT_MODE':
      return {
        ...state,
        session: {
          ...state.session,
          currentMode: action.mode,
        },
      };
    case 'SET_SELECTED_TEXT':
      return {
        ...state,
        session: {
          ...state.session,
          selectedText: action.selectedText,
        },
      };
    case 'CLEAR_MESSAGES':
      return {
        ...state,
        session: {
          ...state.session,
          messages: [],
        },
      };
    case 'RESTORE_SESSION':
      return {
        ...state,
        session: action.session,
      };
    case 'START_NEW_SESSION':
      return {
        ...state,
        session: {
          id: `session_${Date.now()}`,
          messages: [],
          currentMode: 'rag',
          isActive: true,
        },
      };
    default:
      return state;
  }
};

// Create context
interface ChatContextType {
  state: ChatState;
  dispatch: React.Dispatch<ChatAction>;
  startNewSession: () => void;
  addMessage: (message: ChatMessage) => void;
  setVisible: (visible: boolean) => void;
  setMinimized: (minimized: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | undefined) => void;
  setCurrentMode: (mode: 'rag' | 'selection') => void;
  setSelectedText: (text: string | undefined) => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

// Provider component
export const ChatProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(chatReducer, initialState);

  // Load saved conversation from localStorage on mount
  useEffect(() => {
    const savedConversation = localStorage.getItem('docusaurus-chat-session');
    if (savedConversation) {
      try {
        const parsed: any = JSON.parse(savedConversation);
        // Deserialize dates
        const session: ChatSession = {
          ...parsed,
          messages: parsed.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          })),
        };
        dispatch({ type: 'RESTORE_SESSION', session });
        dispatch({ type: 'SET_VISIBLE', isVisible: true });
      } catch (error) {
        console.error('Failed to restore chat session:', error);
        // Start fresh if restoration fails
        dispatch({ type: 'START_NEW_SESSION' });
      }
    }
  }, []);

  // Save conversation to localStorage when it changes
  useEffect(() => {
    if (state.session.messages.length > 0) {
      const serialized = {
        ...state.session,
        messages: state.session.messages.map(msg => ({
          ...msg,
          timestamp: msg.timestamp.toISOString(),
        })),
      };
      localStorage.setItem('docusaurus-chat-session', JSON.stringify(serialized));
    }
  }, [state.session]);

  // Helper functions
  const startNewSession = () => {
    dispatch({ type: 'START_NEW_SESSION' });
  };

  const addMessage = (message: ChatMessage) => {
    dispatch({ type: 'ADD_MESSAGE', message });
  };

  const setVisible = (visible: boolean) => {
    dispatch({ type: 'SET_VISIBLE', isVisible: visible });
  };

  const setMinimized = (minimized: boolean) => {
    dispatch({ type: 'SET_MINIMIZED', isMinimized: minimized });
  };

  const setLoading = (loading: boolean) => {
    dispatch({ type: 'SET_LOADING', isLoading: loading });
  };

  const setError = (error: string | undefined) => {
    dispatch({ type: 'SET_ERROR', error });
  };

  const setCurrentMode = (mode: 'rag' | 'selection') => {
    dispatch({ type: 'SET_CURRENT_MODE', mode });
  };

  const setSelectedText = (text: string | undefined) => {
    dispatch({ type: 'SET_SELECTED_TEXT', selectedText: text });
  };

  const contextValue = {
    state,
    dispatch,
    startNewSession,
    addMessage,
    setVisible,
    setMinimized,
    setLoading,
    setError,
    setCurrentMode,
    setSelectedText,
  };

  return <ChatContext.Provider value={contextValue}>{children}</ChatContext.Provider>;
};

// Custom hook to use the chat context
export const useChat = (): ChatContextType => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};