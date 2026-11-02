package com.smarthealth.platform.controller;

import com.smarthealth.platform.dto.TrendAnalysisResponse;
import com.smarthealth.platform.service.TrendAnalysisService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/trends")
@Tag(name = "Trend Analysis", description = "Context-aware personal deviation score, trend direction, and persistence analysis")
public class TrendAnalysisController {

    private final TrendAnalysisService trendAnalysisService;

    public TrendAnalysisController(TrendAnalysisService trendAnalysisService) {
        this.trendAnalysisService = trendAnalysisService;
    }

    @PostMapping("/patient/{patientId}/{metricType}/analyze")
    @Operation(summary = "Analyze trend, deviation score, and persistence for a patient metric")
    public ResponseEntity<TrendAnalysisResponse> analyzeTrend(
            @PathVariable Long patientId,
            @PathVariable String metricType) {
        TrendAnalysisResponse response = trendAnalysisService.analyzeTrend(patientId, metricType);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/patient/{patientId}")
    @Operation(summary = "Get historical trend analyses for a patient")
    public ResponseEntity<List<TrendAnalysisResponse>> getTrendsByPatient(
            @PathVariable Long patientId) {
        List<TrendAnalysisResponse> response = trendAnalysisService.getTrendsByPatientId(patientId);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/patient/{patientId}/{metricType}")
    @Operation(summary = "Get historical trend analyses for a specific patient and metric")
    public ResponseEntity<List<TrendAnalysisResponse>> getTrendsByPatientAndMetric(
            @PathVariable Long patientId,
            @PathVariable String metricType) {
        List<TrendAnalysisResponse> response = trendAnalysisService.getTrendsByPatientIdAndMetricType(patientId, metricType);
        return ResponseEntity.ok(response);
    }
}
