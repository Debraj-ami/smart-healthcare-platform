package com.smarthealth.platform.controller;

import com.smarthealth.platform.dto.DiseaseAssessmentRequest;
import com.smarthealth.platform.dto.DiseaseAssessmentResponse;
import com.smarthealth.platform.service.AssessmentService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/assessments")
@Tag(name = "Disease Assessments", description = "Disease risk assessment records")
public class AssessmentController {

    private final AssessmentService assessmentService;

    public AssessmentController(AssessmentService assessmentService) {
        this.assessmentService = assessmentService;
    }

    @PostMapping
    @Operation(summary = "Save a disease risk assessment record")
    public ResponseEntity<DiseaseAssessmentResponse> recordAssessment(
            @Valid @RequestBody DiseaseAssessmentRequest request) {
        DiseaseAssessmentResponse response = assessmentService.recordAssessment(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @GetMapping("/patient/{patientId}")
    @Operation(summary = "Get historical assessments for a patient")
    public ResponseEntity<List<DiseaseAssessmentResponse>> getAssessmentsByPatient(
            @PathVariable Long patientId) {
        List<DiseaseAssessmentResponse> response = assessmentService.getAssessmentsByPatientId(patientId);
        return ResponseEntity.ok(response);
    }
}
