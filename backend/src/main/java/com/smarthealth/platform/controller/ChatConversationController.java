package com.smarthealth.platform.controller;

import com.smarthealth.platform.dto.ChatConversationRequest;
import com.smarthealth.platform.dto.ChatConversationResponse;
import com.smarthealth.platform.service.ChatConversationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/chat/conversations")
@Tag(name = "Chatbot Conversations", description = "Chatbot conversation history persistence")
public class ChatConversationController {

    private final ChatConversationService chatConversationService;

    public ChatConversationController(ChatConversationService chatConversationService) {
        this.chatConversationService = chatConversationService;
    }

    @PostMapping
    @Operation(summary = "Log a message in a chatbot conversation session")
    public ResponseEntity<ChatConversationResponse> saveConversation(
            @Valid @RequestBody ChatConversationRequest request) {
        ChatConversationResponse response = chatConversationService.saveConversation(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @GetMapping("/{sessionId}")
    @Operation(summary = "Get conversation logs for a given session ID")
    public ResponseEntity<List<ChatConversationResponse>> getConversationsBySession(
            @PathVariable String sessionId) {
        List<ChatConversationResponse> response = chatConversationService.getConversationsBySessionId(sessionId);
        return ResponseEntity.ok(response);
    }
}
