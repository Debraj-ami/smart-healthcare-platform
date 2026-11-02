import axios from 'axios';
import { ChatRequest, ChatResponse } from '../types/chatbot';

export const CHATBOT_API_URL = import.meta.env.VITE_CHATBOT_API_URL || 'http://localhost:8000';
export const CHATBOT_TIMEOUT_MS = Number(import.meta.env.VITE_CHATBOT_TIMEOUT_MS) || 75000;

const chatbotClient = axios.create({
  baseURL: CHATBOT_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: CHATBOT_TIMEOUT_MS,
});

export const chatbotService = {
  checkHealth: async (): Promise<{ status: string; service: string }> => {
    const response = await chatbotClient.get<{ status: string; service: string }>('/health');
    return response.data;
  },

  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await chatbotClient.post<ChatResponse>('/api/chat', request);
    return response.data;
  },

  sendMessageStream: async (
    request: ChatRequest,
    onToken: (token: string) => void,
    onComplete: (response: ChatResponse) => void,
    onError: (error: any) => void
  ): Promise<void> => {
    try {
      const response = await fetch(`${CHATBOT_API_URL}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`Chatbot service returned HTTP ${response.status}`);
      }

      if (!response.body) {
        throw new Error('Streaming not supported by browser response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() || '';

        for (const block of blocks) {
          if (!block.trim()) continue;
          let eventType = 'message';
          let dataStr = '';

          for (const line of block.split('\n')) {
            const trimmedLine = line.trim();
            if (trimmedLine.startsWith('event:')) {
              eventType = trimmedLine.replace('event:', '').trim();
            } else if (trimmedLine.startsWith('data:')) {
              dataStr = trimmedLine.replace('data:', '').trim();
            }
          }

          if (eventType === 'token' && dataStr) {
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.text) {
                onToken(parsed.text);
              }
            } catch {
              // Ignore partial JSON in data token
            }
          } else if (eventType === 'complete' && dataStr) {
            try {
              const finalResp: ChatResponse = JSON.parse(dataStr);
              onComplete(finalResp);
            } catch (e) {
              console.error('Failed to parse final SSE payload:', e);
            }
          } else if (eventType === 'error' && dataStr) {
            try {
              const errData = JSON.parse(dataStr);
              throw new Error(errData.detail || 'Streaming generation interrupted.');
            } catch (e: any) {
              throw new Error(e.message || 'Streaming generation interrupted.');
            }
          }
        }
      }
    } catch (err: any) {
      onError(err);
    }
  },
};

export default chatbotService;
