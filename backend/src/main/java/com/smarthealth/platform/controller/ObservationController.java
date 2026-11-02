package com.smarthealth.platform.controller;

import com.smarthealth.platform.dto.HealthObservationRequest;
import com.smarthealth.platform.dto.HealthObservationResponse;
import com.smarthealth.platform.service.ObservationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/observations")
@Tag(name = "Health Observations", description = "Time-series clinical and biometric observations")
public class ObservationController {

    private final ObservationService observationService;

    public ObservationController(ObservationService observationService) {
        this.observationService = observationService;
    }

    @PostMapping
    @Operation(summary = "Record a new health observation for a patient")
    public ResponseEntity<HealthObservationResponse> recordObservation(
            @Valid @RequestBody HealthObservationRequest request) {
        HealthObservationResponse response = observationService.recordObservation(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @GetMapping("/patient/{patientId}")
    @Operation(summary = "Get historical health observations for a patient")
    public ResponseEntity<List<HealthObservationResponse>> getObservationsByPatient(
            @PathVariable Long patientId) {
        List<HealthObservationResponse> response = observationService.getObservationsByPatientId(patientId);
        return ResponseEntity.ok(response);
    }
}
