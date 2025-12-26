// Floating chat button component that appears on all pages
import React from 'react';
import { useChat } from './ChatContext';

export const FloatingButton: React.FC = () => {
  const { state, setVisible, setMinimized } = useChat();

  const handleClick = () => {
    if (!state.isVisible) {
      // Show and expand the chat window
      setVisible(true);
      setMinimized(false);
    } else {
      // Toggle minimized state if already visible
      setMinimized(!state.isMinimized);
    }
  };

  if (state.isVisible && !state.isMinimized) {
    // Don't show the button when chat is fully visible
    return null;
  }

  return (
    <button
      onClick={handleClick}
      aria-label={state.isVisible ? "Expand chat" : "Open chat"}
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        width: '60px',
        height: '60px',
        borderRadius: '50%',
        backgroundColor: '#22D3EE',
        color: '#070A12',
        border: 'none',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '1.5rem',
        fontWeight: 'bold',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        zIndex: 999,
        transition: 'all 0.2s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'scale(1.1)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'scale(1)';
      }}
    >
      {state.isVisible && state.isMinimized ? '💬' : '🤖'}
    </button>
  );
};