// Individual message display component
import React from 'react';
import { ChatMessage, Citation } from './types';

interface MessageBubbleProps {
  message: ChatMessage;
  onCitationClick?: (citation: Citation) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message, onCitationClick }) => {
  // Function to render citations if they exist
  const renderCitations = () => {
    if (!message.content.includes('[') || !onCitationClick) return null;

    // Simple regex to find citation patterns like [doc_id, chunk_id]
    const citationRegex = /\[(MODULE-[^\]]+)\]/g;
    const parts = message.content.split(citationRegex);

    return parts.map((part, index) => {
      if (index % 2 === 1) { // This is a citation part
        const [docId, chunkIdStr] = part.split(', ');
        const chunkId = parseInt(chunkIdStr) || 1;

        const citation: Citation = {
          doc_id: docId.trim(),
          chunk_id: chunkId
        };

        return (
          <button
            key={index}
            onClick={() => onCitationClick(citation)}
            className="citation-link"
            style={{
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              border: 'none',
              borderRadius: '4px',
              padding: '2px 4px',
              margin: '0 2px',
              cursor: 'pointer',
              fontSize: '0.8em',
              color: '#22D3EE', // Use accent color
            }}
          >
            [{docId}]
          </button>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <div
      className={`message-bubble ${message.role}`}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
        marginBottom: '1rem',
        maxWidth: '80%',
        alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
      }}
    >
      <div
        style={{
          backgroundColor: message.role === 'user' ? '#0B1220' : '#111827',
          color: '#EAF0FF',
          padding: '0.75rem 1rem',
          borderRadius: message.role === 'user'
            ? '16px 4px 16px 16px'
            : '4px 16px 16px 16px',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          minWidth: '100px',
          maxWidth: '100%',
          wordWrap: 'break-word',
          whiteSpace: 'pre-wrap',
        }}
      >
        {message.isLoading ? (
          <div className="loading-indicator">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
        ) : (
          <>
            {renderCitations()}
            {!renderCitations() && message.content}
          </>
        )}
      </div>
      <div
        style={{
          fontSize: '0.75rem',
          color: 'rgba(234, 240, 255, 0.72)',
          marginTop: '0.25rem',
          paddingLeft: message.role === 'user' ? '1rem' : '0',
          paddingRight: message.role === 'user' ? '0' : '1rem',
        }}
      >
        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </div>
    </div>
  );
};