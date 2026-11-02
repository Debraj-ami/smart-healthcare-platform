import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { ChatAssistantContextData, ChatMessage } from '../types/chatbot';
import { usePatient } from './PatientContext';

interface ChatAssistantContextType {
  isOpen: boolean;
  activeContext: ChatAssistantContextData | null;
  pageContext: ChatAssistantContextData | null;
  openChat: (context?: ChatAssistantContextData) => void;
  closeChat: () => void;
  toggleChat: () => void;
  setPageContext: (context: ChatAssistantContextData | null) => void;
  clearContext: () => void;
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  sessionId: string;
  resetConversation: () => void;
}

const ChatAssistantContext = createContext<ChatAssistantContextType | undefined>(undefined);

export const ChatAssistantProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { activePatientId } = usePatient();
  const [isOpen, setIsOpen] = useState(false);
  const [activeContext, setActiveContext] = useState<ChatAssistantContextData | null>(null);
  const [pageContext, setPageContextState] = useState<ChatAssistantContextData | null>(null);
  const [sessionId, setSessionId] = useState<string>(() => `session_${Date.now()}`);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  // When patient switches, reset conversation and context to ensure strict cross-patient isolation
  useEffect(() => {
    setActiveContext(null);
    setSessionId(`session_patient_${activePatientId}_${Date.now()}`);
    setMessages([]);
  }, [activePatientId]);

  const openChat = useCallback((context?: ChatAssistantContextData) => {
    if (context) {
      setActiveContext(context);
    } else if (pageContext) {
      setActiveContext(pageContext);
    }
    setIsOpen(true);
  }, [pageContext]);

  const closeChat = useCallback(() => {
    setIsOpen(false);
  }, []);

  const toggleChat = useCallback(() => {
    setIsOpen((prev) => {
      if (!prev && !activeContext && pageContext) {
        setActiveContext(pageContext);
      }
      return !prev;
    });
  }, [activeContext, pageContext]);

  const setPageContext = useCallback((context: ChatAssistantContextData | null) => {
    setPageContextState(context);
  }, []);

  const clearContext = useCallback(() => {
    setActiveContext(null);
  }, []);

  const resetConversation = useCallback(() => {
    setSessionId(`session_patient_${activePatientId}_${Date.now()}`);
    setMessages([]);
  }, [activePatientId]);

  return (
    <ChatAssistantContext.Provider
      value={{
        isOpen,
        activeContext,
        pageContext,
        openChat,
        closeChat,
        toggleChat,
        setPageContext,
        clearContext,
        messages,
        setMessages,
        sessionId,
        resetConversation,
      }}
    >
      {children}
    </ChatAssistantContext.Provider>
  );
};

export const useChatAssistant = (): ChatAssistantContextType | undefined => {
  return useContext(ChatAssistantContext);
};

export default ChatAssistantContext;
