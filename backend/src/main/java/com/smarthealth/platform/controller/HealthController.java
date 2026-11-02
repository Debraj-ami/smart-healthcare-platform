package com.smarthealth.platform.controller;

import com.smarthealth.platform.dto.HealthCheckResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/health")
@Tag(name = "Health Check", description = "Service health and readiness check")
public class HealthController {

    @GetMapping
    @Operation(summary = "Check backend service health status")
    public ResponseEntity<HealthCheckResponse> checkHealth() {
        return ResponseEntity.ok(new HealthCheckResponse("healthy", "smart-healthcare-backend"));
    }
}
