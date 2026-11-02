package com.smarthealth.platform.dto;

import com.smarthealth.platform.entity.MessageRole;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

public class ChatConversationRequest {

    private Long userId;

    @NotBlank(message = "Session ID is required")
    private String sessionId;

    @NotNull(message = "Message role is required")
    private MessageRole messageRole;

    @NotBlank(message = "Content is required")
    private String content;

    public ChatConversationRequest() {
    }

    public ChatConversationRequest(Long userId, String sessionId, MessageRole messageRole, String content) {
        this.userId = userId;
        this.sessionId = sessionId;
        this.messageRole = messageRole;
        this.content = content;
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
}
