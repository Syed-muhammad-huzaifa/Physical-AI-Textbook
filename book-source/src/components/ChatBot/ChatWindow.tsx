// Chat window with messages component
import React, { useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';
import { TextInput } from './TextInput';
import { useChat } from './ChatContext';
import { sendChatRequest, checkBackendHealth } from './api';
import { ChatMessage, Citation } from './types';

export const ChatWindow: React.FC = () => {
  const { state, addMessage, setLoading, setError, setCurrentMode, setSelectedText } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom of messages when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [state.session.messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (message: string) => {
    try {
      // Add user message to UI immediately
      const userMessage: ChatMessage = {
        id: `msg-${Date.now()}-${Math.random()}`,
        role: 'user',
        content: message,
        timestamp: new Date(),
      };
      addMessage(userMessage);

      // Set loading state
      setLoading(true);

      // Prepare the chat request based on current mode
      const request = {
        query: message,
        mode: state.session.currentMode,
        selected_text: state.session.currentMode === 'selection' ? state.session.selectedText : undefined,
        top_k: 6, // Default value
      };

      // Send the request to the backend
      const response = await sendChatRequest(request);

      // Create assistant message with response
      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now()}-${Math.random()}`,
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
      };

      addMessage(assistantMessage);

      // Clear selection after successful response in selection mode
      if (state.session.currentMode === 'selection') {
        setTimeout(() => {
          setCurrentMode('rag');
          setSelectedText(undefined);
        }, 1000);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setError(error instanceof Error ? error.message : 'Failed to send message');

      // Add error message to chat
      const errorMessage: ChatMessage = {
        id: `msg-error-${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      addMessage(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleCitationClick = (citation: Citation) => {
    // In the future, this could navigate to the specific document/chunk
    console.log('Citation clicked:', citation);
    alert(`Would navigate to document: ${citation.doc_id}, chunk: ${citation.chunk_id}`);
  };

  const toggleMinimize = () => {
    // This would be handled by the parent component
  };

  return (
    <div
      style={{
        display: state.isVisible ? 'flex' : 'none',
        flexDirection: 'column',
        width: '400px',
        height: state.isMinimized ? '400px' : '600px',
        backgroundColor: '#070A12',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '16px',
        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.5)',
        overflow: 'hidden',
        color: '#EAF0FF',
        zIndex: 1000,
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        fontFamily: 'Inter, sans-serif',
      }}
    >
      {/* Chat Header */}
      <div
        style={{
          backgroundColor: '#0B1220',
          padding: '1rem',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div>
          <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '600' }}>Robotics Assistant</h3>
          <p style={{ margin: '0.25rem 0 0', fontSize: '0.8rem', opacity: 0.7 }}>
            {state.session.currentMode === 'selection' ? 'Selection Mode' : 'Knowledge Base Mode'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            onClick={toggleMinimize}
            style={{
              background: 'none',
              border: 'none',
              color: '#EAF0FF',
              cursor: 'pointer',
              fontSize: '1.2rem',
              padding: '0.25rem',
            }}
          >
            {state.isMinimized ? '+' : '−'}
          </button>
          <button
            onClick={() => {
              // Close chat functionality would be handled by parent
            }}
            style={{
              background: 'none',
              border: 'none',
              color: '#EAF0FF',
              cursor: 'pointer',
              fontSize: '1.2rem',
              padding: '0.25rem',
            }}
          >
            ×
          </button>
        </div>
      </div>

      {/* Messages Container */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '1rem',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: '#070A12',
        }}
      >
        {state.session.messages.length === 0 ? (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              textAlign: 'center',
              color: 'rgba(234, 240, 255, 0.6)',
            }}
          >
            <h4 style={{ marginBottom: '0.5rem', fontWeight: '500' }}>Welcome to Robotics Assistant!</h4>
            <p style={{ fontSize: '0.9rem', maxWidth: '80%' }}>
              {state.session.currentMode === 'selection' && state.session.selectedText
                ? 'Ask questions about the selected text'
                : 'Ask me anything about robotics and humanoid systems'}
            </p>
          </div>
        ) : (
          state.session.messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              onCitationClick={handleCitationClick}
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <TextInput
        onSendMessage={handleSendMessage}
        disabled={state.isLoading}
      />

      {/* Loading Indicator */}
      {state.isLoading && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            backgroundColor: 'rgba(7, 10, 18, 0.8)',
            padding: '1rem',
            borderRadius: '8px',
            zIndex: 1001,
          }}
        >
          <div className="loading-indicator">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {state.error && (
        <div
          style={{
            position: 'absolute',
            bottom: '60px',
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: '#EF4444',
            color: 'white',
            padding: '0.5rem 1rem',
            borderRadius: '4px',
            zIndex: 1001,
            fontSize: '0.8rem',
          }}
        >
          {state.error}
        </div>
      )}
    </div>
  );
};