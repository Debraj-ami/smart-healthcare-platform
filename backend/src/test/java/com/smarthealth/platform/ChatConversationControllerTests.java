package com.smarthealth.platform;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.smarthealth.platform.dto.ChatConversationRequest;
import com.smarthealth.platform.entity.MessageRole;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

import java.util.UUID;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class ChatConversationControllerTests {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void shouldPersistAndRetrieveChatConversation() throws Exception {
        String sessionId = "sess-" + UUID.randomUUID();

        ChatConversationRequest userMsg = new ChatConversationRequest(
                null, sessionId, MessageRole.USER, "I have mild abdominal discomfort and fatigue."
        );

        mockMvc.perform(post("/api/chat/conversations")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(userMsg)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").isNumber())
                .andExpect(jsonPath("$.sessionId").value(sessionId))
                .andExpect(jsonPath("$.messageRole").value("USER"))
                .andExpect(jsonPath("$.content").value("I have mild abdominal discomfort and fatigue."));

        ChatConversationRequest botMsg = new ChatConversationRequest(
                null, sessionId, MessageRole.ASSISTANT,
                "Thank you for sharing your symptoms. Please note that this information is educational."
        );

        mockMvc.perform(post("/api/chat/conversations")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(botMsg)))
                .andExpect(status().isCreated());

        mockMvc.perform(get("/api/chat/conversations/" + sessionId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray())
                .andExpect(jsonPath("$.length()").value(2))
                .andExpect(jsonPath("$[0].messageRole").value("USER"))
                .andExpect(jsonPath("$[1].messageRole").value("ASSISTANT"));
    }
}
