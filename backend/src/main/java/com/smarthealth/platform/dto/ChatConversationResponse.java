package com.smarthealth.platform.dto;

import com.smarthealth.platform.entity.ChatbotConversation;
import com.smarthealth.platform.entity.MessageRole;

import java.time.LocalDateTime;

public class ChatConversationResponse {

    private Long id;
    private Long userId;
    private String sessionId;
    private MessageRole messageRole;
    private String content;
    private LocalDateTime timestamp;

    public ChatConversationResponse() {
    }

    public static ChatConversationResponse fromEntity(ChatbotConversation conversation) {
        ChatConversationResponse response = new ChatConversationResponse();
        response.setId(conversation.getId());
        response.setUserId(conversation.getUser() != null ? conversation.getUser().getId() : null);
        response.setSessionId(conversation.getSessionId());
        response.setMessageRole(conversation.getMessageRole());
        response.setContent(conversation.getContent());
        response.setTimestamp(conversation.getTimestamp());
        return response;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getUserId() {
        return userId;
    }

    public void setUserId(Long userId) {
        this.userId = userId;
    }

    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    public MessageRole getMessageRole() {
        return messageRole;
    }

    public void setMessageRole(MessageRole messageRole) {
        this.messageRole = messageRole;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
}
