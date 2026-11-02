import React from 'react';
import { useChatAssistant } from '../../context/ChatAssistantContext';
import { usePatient } from '../../context/PatientContext';
import ChatWindow from './ChatWindow';

export const FloatingChatDrawer: React.FC = () => {
  const { isOpen, closeChat, activeContext } = useChatAssistant();
  const { activePatientId } = usePatient();

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="false"
      aria-label="Smart Health AI Assistant"
      className="fixed bottom-4 sm:bottom-6 right-3 sm:right-6 z-[60] w-[95vw] sm:w-[440px] h-[600px] sm:h-[650px] max-h-[calc(100vh-3rem)] rounded-2xl shadow-2xl border border-slate-300/80 bg-white flex flex-col overflow-hidden animate-in fade-in slide-in-from-bottom-6 duration-200"
    >
      <ChatWindow
        patientId={activePatientId}
        context={activeContext || undefined}
        onClose={closeChat}
        isDrawer={true}
      />
    </div>
  );
};

export default FloatingChatDrawer;
