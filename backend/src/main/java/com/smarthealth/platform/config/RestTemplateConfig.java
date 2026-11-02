package com.smarthealth.platform.config;

import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;

/**
 * Provides a shared RestTemplate bean for outbound HTTP calls to microservices
 * (primarily the Python ML inference service).
 *
 * Timeouts are set conservatively for ML inference which may involve
 * model loading on first request:
 *   - Connect timeout: 5 seconds
 *   - Read timeout:    30 seconds (allows for slower first inference)
 */
@Configuration
public class RestTemplateConfig {

    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder builder) {
        return builder
                .connectTimeout(Duration.ofSeconds(5))
                .readTimeout(Duration.ofSeconds(30))
                .build();
    }
}
