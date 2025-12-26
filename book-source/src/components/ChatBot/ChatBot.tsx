// Main chatbot component that integrates all functionality
import React from 'react';
import { ChatWindow } from './ChatWindow';
import { FloatingButton } from './FloatingButton';
import { useChat } from './ChatContext';

export const ChatBot: React.FC = () => {
  const { state } = useChat();

  return (
    <>
      {state.isVisible && <ChatWindow />}
      <FloatingButton />
    </>
  );
};