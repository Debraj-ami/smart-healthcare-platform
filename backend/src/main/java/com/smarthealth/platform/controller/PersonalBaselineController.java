package com.smarthealth.platform.controller;

import com.smarthealth.platform.dto.PersonalBaselineResponse;
import com.smarthealth.platform.service.PersonalBaselineService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/baselines")
@Tag(name = "Personal Baselines", description = "Individualized patient baseline calculation and monitoring (statistical personalization)")
public class PersonalBaselineController {

    private final PersonalBaselineService personalBaselineService;

    public PersonalBaselineController(PersonalBaselineService personalBaselineService) {
        this.personalBaselineService = personalBaselineService;
    }

    @PostMapping("/patient/{patientId}/{metricType}/calculate")
    @Operation(summary = "Explicitly calculate or recalculate a patient baseline for a metric")
    public ResponseEntity<PersonalBaselineResponse> calculateBaseline(
            @PathVariable Long patientId,
            @PathVariable String metricType) {
        PersonalBaselineResponse response = personalBaselineService.calculateBaseline(patientId, metricType);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/patient/{patientId}")
    @Operation(summary = "Get all available baselines for a patient")
    public ResponseEntity<List<PersonalBaselineResponse>> getBaselinesByPatient(
            @PathVariable Long patientId) {
        List<PersonalBaselineResponse> response = personalBaselineService.getBaselinesByPatientId(patientId);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/patient/{patientId}/{metricType}")
    @Operation(summary = "Get the baseline for a specific patient and metric")
    public ResponseEntity<PersonalBaselineResponse> getBaseline(
            @PathVariable Long patientId,
            @PathVariable String metricType) {
        PersonalBaselineResponse response = personalBaselineService.getBaseline(patientId, metricType);
        return ResponseEntity.ok(response);
    }
}
