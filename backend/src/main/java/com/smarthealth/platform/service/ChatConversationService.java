package com.smarthealth.platform.service;

import com.smarthealth.platform.dto.ChatConversationRequest;
import com.smarthealth.platform.dto.ChatConversationResponse;
import com.smarthealth.platform.entity.ChatbotConversation;
import com.smarthealth.platform.entity.User;
import com.smarthealth.platform.repository.ChatbotConversationRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class ChatConversationService {

    private final ChatbotConversationRepository conversationRepository;
    private final UserService userService;

    public ChatConversationService(ChatbotConversationRepository conversationRepository, UserService userService) {
        this.conversationRepository = conversationRepository;
        this.userService = userService;
    }

    @Transactional
    public ChatConversationResponse saveConversation(ChatConversationRequest request) {
        User user = null;
        if (request.getUserId() != null) {
            user = userService.getUserEntity(request.getUserId());
        }

        ChatbotConversation conversation = new ChatbotConversation(
                user,
                request.getSessionId(),
                request.getMessageRole(),
                request.getContent()
        );

        ChatbotConversation saved = conversationRepository.save(conversation);
        return ChatConversationResponse.fromEntity(saved);
    }

    @Transactional(readOnly = true)
    public List<ChatConversationResponse> getConversationsBySessionId(String sessionId) {
        return conversationRepository.findBySessionIdOrderByTimestampAsc(sessionId)
                .stream()
                .map(ChatConversationResponse::fromEntity)
                .collect(Collectors.toList());
    }
}
