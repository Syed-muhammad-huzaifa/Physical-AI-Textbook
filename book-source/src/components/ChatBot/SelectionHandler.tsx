// Text selection detection and handling functionality
import React, { useEffect, useCallback } from 'react';
import { useChat } from './ChatContext';
import { TextSelection } from './types';

interface SelectionHandlerProps {
  children: React.ReactNode;
}

/**
 * Component that handles text selection detection across the document
 */
export const SelectionHandler: React.FC<SelectionHandlerProps> = ({ children }) => {
  const { setCurrentMode, setSelectedText } = useChat();

  /**
   * Handle text selection event
   */
  const handleSelection = useCallback(() => {
    const selection = window.getSelection();
    if (!selection) return;

    const selectedText = selection.toString().trim();
    if (selectedText.length === 0) {
      // No text selected, return to RAG mode
      setCurrentMode('rag');
      setSelectedText(undefined);
      return;
    }

    // If text is selected, switch to selection mode
    if (selectedText.length > 0) {
      const textSelection: TextSelection = {
        text: selectedText,
        elementId: selection.anchorNode?.parentElement?.id || undefined,
        pageUrl: window.location.href,
        timestamp: new Date(),
      };

      setCurrentMode('selection');
      setSelectedText(selectedText);

      // Optional: Log the selection for analytics or debugging
      console.log('Text selected for chat:', textSelection);
    }
  }, [setCurrentMode, setSelectedText]);

  useEffect(() => {
    // Add event listeners for text selection
    const handleMouseUp = () => {
      setTimeout(handleSelection, 0); // Delay to allow selection to complete
    };

    const handleKeyUp = (event: Event) => {
      if (event instanceof KeyboardEvent && event.key === 'Escape') {
        // Clear selection when Escape is pressed
        setCurrentMode('rag');
        setSelectedText(undefined);
      }
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('keyup', handleKeyUp);

    // Cleanup event listeners
    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('keyup', handleKeyUp);
    };
  }, [handleSelection, setCurrentMode, setSelectedText]);

  return <>{children}</>;
};