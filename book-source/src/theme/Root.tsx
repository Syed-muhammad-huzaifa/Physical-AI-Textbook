// Wrap the app with chatbot functionality
import React from 'react';
import {ChatProvider} from '../components/ChatBot/ChatContext';
import {SelectionHandler} from '../components/ChatBot/SelectionHandler';
import {ChatBot} from '../components/ChatBot/ChatBot';

interface Props {
  children: React.ReactNode;
}

export default function Root({children}: Props) {
  return (
    <ChatProvider>
      <SelectionHandler>
        {children}
        <ChatBot />
      </SelectionHandler>
    </ChatProvider>
  );
}
