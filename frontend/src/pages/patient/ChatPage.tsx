import React from 'react';
import { usePatient } from '../../context/PatientContext';
import ChatWindow from '../../components/chatbot/ChatWindow';
import DisclaimerAlert from '../../components/common/DisclaimerAlert';
import { ShieldCheck, Sparkles } from 'lucide-react';

export const ChatPage: React.FC = () => {
  const { activePatientId } = usePatient();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2">
            <span>AI Health Assistant</span>
            <Sparkles className="w-5 h-5 text-teal-600" />
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Conversational clinical inquiry and symptom extraction for Patient #{activePatientId}.
          </p>
        </div>
      </div>

      <DisclaimerAlert variant="warning" />

      {/* Main Chat Interface */}
      <ChatWindow patientId={activePatientId} />
    </div>
  );
};

export default ChatPage;
