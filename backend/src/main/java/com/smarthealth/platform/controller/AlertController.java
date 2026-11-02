package com.smarthealth.platform.controller;

import com.smarthealth.platform.dto.AlertResponse;
import com.smarthealth.platform.service.AlertService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/alerts")
@Tag(name = "Alerts", description = "Patient alerts and clinical risk notifications")
public class AlertController {

    private final AlertService alertService;

    public AlertController(AlertService alertService) {
        this.alertService = alertService;
    }

    @GetMapping("/patient/{patientId}")
    @Operation(summary = "Get all alerts for a patient")
    public ResponseEntity<List<AlertResponse>> getAlertsByPatient(@PathVariable Long patientId) {
        List<AlertResponse> alerts = alertService.getAlertsByPatientId(patientId);
        return ResponseEntity.ok(alerts);
    }
}
