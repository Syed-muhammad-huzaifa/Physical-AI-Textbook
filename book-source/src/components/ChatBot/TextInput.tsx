// Input field with send button component
import React, { useState, useRef, KeyboardEvent, ChangeEvent } from 'react';
import { useChat } from './ChatContext';
import { ChatMessage } from './types';

interface TextInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

export const TextInput: React.FC<TextInputProps> = ({ onSendMessage, disabled = false }) => {
  const [inputValue, setInputValue] = useState('');
  const { state } = useChat();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    const trimmedValue = inputValue.trim();
    if (trimmedValue && !disabled) {
      onSendMessage(trimmedValue);
      setInputValue('');
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // Prevent new line
      handleSubmit();
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);

    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-end',
        gap: '0.5rem',
        padding: '1rem',
        backgroundColor: '#0B1220',
        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
      }}
    >
      <textarea
        ref={textareaRef}
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder={
          state.session.currentMode === 'selection' && state.session.selectedText
            ? 'Ask about the selected text...'
            : 'Ask about the robotics textbook...'
        }
        disabled={disabled}
        rows={1}
        style={{
          flex: 1,
          minHeight: '40px',
          maxHeight: '120px',
          padding: '0.75rem',
          borderRadius: '8px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          backgroundColor: '#111827',
          color: '#EAF0FF',
          resize: 'none',
          outline: 'none',
          fontFamily: 'inherit',
          fontSize: '1rem',
        }}
      />
      <button
        onClick={handleSubmit}
        disabled={!inputValue.trim() || disabled}
        style={{
          backgroundColor: inputValue.trim() && !disabled ? '#22D3EE' : 'rgba(34, 211, 238, 0.3)',
          color: '#070A12',
          border: 'none',
          borderRadius: '8px',
          padding: '0.75rem 1rem',
          cursor: inputValue.trim() && !disabled ? 'pointer' : 'not-allowed',
          fontWeight: 'bold',
          transition: 'background-color 0.2s',
        }}
      >
        Send
      </button>
    </div>
  );
};