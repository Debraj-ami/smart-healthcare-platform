import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, AlertCircle, RefreshCw, Sparkles, X, ChevronRight } from 'lucide-react';
import { ChatMessage, ChatResponse, ChatAssistantContextData } from '../../types/chatbot';
import chatbotService from '../../services/chatbotService';
import MessageBubble from './MessageBubble';
import Button from '../common/Button';
import { useChatAssistant } from '../../context/ChatAssistantContext';
import { formatMetricName } from '../../utils/formatters';

interface ChatWindowProps {
  patientId: number;
  context?: ChatAssistantContextData;
  onClose?: () => void;
  isDrawer?: boolean;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  patientId,
  context: propContext,
  onClose,
  isDrawer = false,
}) => {
  // Use shared assistant context if mounted inside ChatAssistantProvider
  const assistantContext = useChatAssistant();

  // Fallback local states if outside provider
  const [localMessages, setLocalMessages] = useState<ChatMessage[]>([]);
  const [localSessionId] = useState<string>(() => `session_${Date.now()}`);

  const messages = assistantContext ? assistantContext.messages : localMessages;
  const setMessages = assistantContext ? assistantContext.setMessages : setLocalMessages;
  const sessionId = assistantContext ? assistantContext.sessionId : localSessionId;
  const activeContext = propContext || (assistantContext ? assistantContext.activeContext : null);

  const [inputMessage, setInputMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isServiceOnline, setIsServiceOnline] = useState<boolean | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Check FastAPI service health on mount
  useEffect(() => {
    const checkService = async () => {
      try {
        const res = await chatbotService.checkHealth();
        setIsServiceOnline(res.status === 'healthy');
      } catch {
        setIsServiceOnline(false);
      }
    };
    checkService();
  }, []);

  // Initial welcome message from clinical assistant
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          id: 'welcome_1',
          sender: 'assistant',
          text: `Hello, I'm your SmartHealth AI clinical assistant. I can help answer general health questions, explore reported symptoms, review your health monitoring, and discuss risk assessments.\n\nPlease note: I do not provide autonomous clinical diagnoses or prescribe treatments. How can I help you today?`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          safetyStatus: 'NORMAL_CONVERSATION',
        },
      ]);
    }
  }, [messages.length, setMessages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const executeSendMessage = async (textToSend: string) => {
    const trimmed = textToSend.trim();
    if (!trimmed || isLoading) return;

    const userMsg: ChatMessage = {
      id: `user_${Date.now()}`,
      sender: 'user',
      text: trimmed,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputMessage('');
    setIsLoading(true);

    const assistantId = `assistant_${Date.now()}`;
    const initialAssistantMsg: ChatMessage = {
      id: assistantId,
      sender: 'assistant',
      text: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      safetyStatus: 'NORMAL_CONVERSATION',
    };

    setMessages((prev) => [...prev, initialAssistantMsg]);

    const requestPayload = {
      message: trimmed,
      session_id: sessionId,
      user_id: patientId.toString(),
      context: activeContext || undefined,
    };

    try {
      await chatbotService.sendMessageStream(
        requestPayload,
        (token: string) => {
          setIsLoading(false);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId ? { ...msg, text: msg.text + token } : msg
            )
          );
        },
        (response: ChatResponse) => {
          setIsLoading(false);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    text: response.assistant_message || msg.text,
                    safetyStatus: response.safety_status,
                    intent: response.intent,
                    structuredData: response.structured_health_information || response.structured_data,
                    currentTurnExtraction: response.current_turn_extraction,
                    assessmentHandoff: response.assessment_handoff || response.handoff_contract,
                  }
                : msg
            )
          );
        },
        async (streamErr: any) => {
          // Fallback to standard non-streaming sendMessage if stream is interrupted or unavailable
          try {
            const fallbackResp = await chatbotService.sendMessage(requestPayload);
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? {
                      ...msg,
                      text: fallbackResp.assistant_message || '',
                      safetyStatus: fallbackResp.safety_status,
                      intent: fallbackResp.intent,
                      structuredData: fallbackResp.structured_health_information || fallbackResp.structured_data,
                      currentTurnExtraction: fallbackResp.current_turn_extraction,
                      assessmentHandoff: fallbackResp.assessment_handoff || fallbackResp.handoff_contract,
                    }
                  : msg
              )
            );
          } catch (fallbackErr: any) {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? {
                      ...msg,
                      text: `Unable to connect to the Chatbot service on port 8000 (${fallbackErr.message || streamErr.message}). Ensure the FastAPI service is running: \`uvicorn app.main:app --port 8000\`.`,
                    }
                  : msg
              )
            );
          } finally {
            setIsLoading(false);
          }
        }
      );
    } catch (err: any) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                text: `Unable to connect to the Chatbot service on port 8000 (${err.message}). Ensure the FastAPI service is running: \`uvicorn app.main:app --port 8000\`.`,
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    executeSendMessage(inputMessage);
  };

  const handleChipClick = (question: string) => {
    executeSendMessage(question);
  };

  // Dynamic suggested quick prompts based on active context
  const getSuggestedPrompts = () => {
    if (!activeContext) {
      return [
        'What does my recent health monitoring show?',
        'How have my health readings been recently?',
        'What should I discuss with my doctor?',
      ];
    }
    if (activeContext.source === 'assessment' || activeContext.assessmentId) {
      return [
        'What does this result mean?',
        'Why did the model give me this result?',
        'Does this mean I have heart disease?',
        'How does this compare with my previous assessment?',
        'What should I discuss with my doctor?',
      ];
    }
    if (activeContext.source === 'monitoring' || activeContext.metric) {
      return [
        `What does my recent ${activeContext.metric ? formatMetricName(activeContext.metric) : 'health'} monitoring show?`,
        'How does my latest reading compare with my baseline?',
        'Is my trend going up or down?',
        'What should I discuss with my doctor?',
      ];
    }
    return [
      'What does my recent health monitoring show?',
      'What should I discuss with my doctor?',
    ];
  };

  const suggestedPrompts = getSuggestedPrompts();

  return (
    <div className={`flex flex-col ${isDrawer ? 'h-full' : 'h-[650px]'} bg-slate-50/50 rounded-2xl border border-slate-200 overflow-hidden shadow-card`}>
      {/* Chat Header */}
      <div className="bg-white px-4 sm:px-6 py-3.5 border-b border-slate-200 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-teal-500/10 border border-teal-500/20 text-teal-600 flex items-center justify-center flex-shrink-0">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800 flex items-center gap-1.5">
              <span>Smart Health AI Assistant</span>
              <span className="text-[10px] bg-teal-50 text-teal-700 font-semibold px-2 py-0.5 rounded-full border border-teal-200">
                Context-Aware
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Grounded clinical decision support & inquiry
            </p>
          </div>
        </div>

        {/* Right Header Actions */}
        <div className="flex items-center gap-2">
          {isServiceOnline === true ? (
            <span className="hidden sm:flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-[11px] font-semibold text-emerald-700">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              Online
            </span>
          ) : isServiceOnline === false ? (
            <span className="hidden sm:flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-amber-50 border border-amber-200 text-[11px] font-semibold text-amber-700">
              <AlertCircle className="w-3 h-3" />
              Offline
            </span>
          ) : null}

          {/* Reset Conversation Button */}
          {assistantContext && (
            <button
              type="button"
              onClick={assistantContext.resetConversation}
              className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
              title="Reset conversation"
              aria-label="Reset conversation"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          )}

          {/* Close/Minimize Button for Drawer */}
          {(isDrawer || onClose) && (
            <button
              type="button"
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
              title="Close Assistant"
              aria-label="Close Assistant"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Contextual Indicator Banner */}
      {activeContext && (
        <div className="bg-gradient-to-r from-teal-50 to-cyan-50 border-b border-teal-200/80 px-4 py-2 flex items-center justify-between text-xs text-teal-900 flex-shrink-0">
          <div className="flex items-center gap-2 truncate">
            <span className="w-2 h-2 rounded-full bg-teal-500 flex-shrink-0 animate-pulse" />
            <span className="font-semibold truncate">
              {activeContext.source === 'assessment' || activeContext.assessmentId
                ? `Discussing: Heart Assessment #${activeContext.assessmentId || ''}`
                : activeContext.source === 'monitoring' && activeContext.metric
                ? `Discussing: ${formatMetricName(activeContext.metric)} Monitoring`
                : `Active Context: ${activeContext.source || 'Patient Health Record'}`}
            </span>
          </div>
          {assistantContext && (
            <button
              type="button"
              onClick={assistantContext.clearContext}
              className="text-[11px] text-teal-700 hover:text-teal-900 underline font-medium flex-shrink-0 ml-2"
            >
              Clear context
            </button>
          )}
        </div>
      )}

      {/* Messages Scroll Area */}
      <div className="flex-1 p-3 sm:p-4 overflow-y-auto">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {isLoading && (
          <div className="flex gap-2.5 items-start my-2.5">
            <div className="w-8 h-8 rounded-xl bg-teal-600 text-white flex items-center justify-center flex-shrink-0">
              <Bot className="w-4 h-4" />
            </div>
            <div className="p-3.5 rounded-2xl bg-white border border-slate-200 text-slate-400 text-xs shadow-card flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-teal-500 animate-bounce"></span>
              <span className="w-2 h-2 rounded-full bg-teal-500 animate-bounce [animation-delay:0.2s]"></span>
              <span className="w-2 h-2 rounded-full bg-teal-500 animate-bounce [animation-delay:0.4s]"></span>
              <span className="text-xs text-slate-500 ml-1">Retrieving authoritative data & reasoning...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Quick Question Chips */}
      <div className="px-3 sm:px-4 py-2 bg-slate-50/90 border-t border-slate-200/70 overflow-x-auto flex items-center gap-1.5 no-scrollbar flex-shrink-0">
        <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider flex-shrink-0 flex items-center gap-1">
          <Sparkles className="w-3 h-3 text-teal-600" />
          Suggested:
        </span>
        {suggestedPrompts.slice(0, 3).map((prompt, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => handleChipClick(prompt)}
            disabled={isLoading}
            className="text-[11px] font-medium px-2.5 py-1 rounded-full bg-white hover:bg-teal-50 border border-slate-200 hover:border-teal-200 text-slate-700 hover:text-teal-800 transition-all whitespace-nowrap shadow-xs disabled:opacity-50"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Input Bar */}
      <form onSubmit={handleSend} className="p-3 sm:p-4 bg-white border-t border-slate-200 flex gap-2 flex-shrink-0">
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder={
            activeContext?.source === 'assessment' || activeContext?.assessmentId
              ? "Ask about this assessment (e.g. 'What does this result mean?')..."
              : "Ask a health question or inquiry..."
          }
          className="flex-1 px-3 sm:px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xs sm:text-sm placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
          disabled={isLoading}
        />
        <Button
          type="submit"
          variant="teal"
          size="md"
          disabled={!inputMessage.trim() || isLoading}
          isLoading={isLoading}
          rightIcon={<Send className="w-4 h-4" />}
        >
          Send
        </Button>
      </form>
    </div>
  );
};

export default ChatWindow;
