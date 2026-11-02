package com.smarthealth.platform.repository;

import com.smarthealth.platform.entity.ChatbotConversation;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ChatbotConversationRepository extends JpaRepository<ChatbotConversation, Long> {
    List<ChatbotConversation> findBySessionIdOrderByTimestampAsc(String sessionId);
}
