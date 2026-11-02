package com.smarthealth.platform.controller;

import com.smarthealth.platform.dto.DiseaseAssessmentResponse;
import com.smarthealth.platform.dto.HeartPredictRequest;
import com.smarthealth.platform.dto.HeartPredictResponse;
import com.smarthealth.platform.service.HeartAssessmentService;
import com.smarthealth.platform.service.MlServiceClient;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * REST controller for Heart Disease ML prediction and orchestration.
 *
 * Endpoint:
 * POST /api/assessments/heart/predict
 *   Accepts validated 13 Cleveland Heart clinical features, calls the Python FastAPI ML microservice,
 *   optionally persists the assessment record if patientId is provided, and returns the full
 *   prediction + SHAP log-odds explanation to the caller.
 *
 * Architecture:
 *   Frontend/Caller -> POST /api/assessments/heart/predict (Spring Boot)
 *                   -> POST /api/predict/heart           (Python FastAPI ML service)
 *                   <- probability + threshold + SHAP explanation
 *                   -> optional persistence to disease_assessments table
 *                   <- HeartPredictResponse
 */
@RestController
@RequestMapping("/api/assessments")
@Tag(name = "ML Predictions", description = "ML inference endpoints integrated with the Python microservice")
public class HeartPredictController {

    private final HeartAssessmentService heartAssessmentService;

    public HeartPredictController(HeartAssessmentService heartAssessmentService) {
        this.heartAssessmentService = heartAssessmentService;
    }

    /**
     * Executes a Heart Disease risk screening assessment using the production Logistic Regression
     * (Robust L1) model and the configured screening threshold (t=0.31).
     *
     * @param request validated 13 Heart clinical feature inputs + optional patientId
     * @return full prediction response with model probability, threshold, and SHAP attributions
     */
    @PostMapping("/heart/predict")
    @Operation(
        summary = "Heart Disease Risk Prediction",
        description = "Runs ML inference for Heart Disease risk using the production Logistic Regression model. " +
                      "Exposes model-estimated probability, binary classification decision against threshold 0.31, " +
                      "and local SHAP feature contributions in log-odds space. No synthetic values are generated."
    )
    public ResponseEntity<?> predictHeart(@Valid @RequestBody HeartPredictRequest request) {
        try {
            HeartPredictResponse response = heartAssessmentService.predict(request);
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
        } catch (MlServiceClient.MlServiceException ex) {
            // Surface actionable error: ML service unavailable or returned an error
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(
                Map.of(
                    "error", "ML service unavailable",
                    "message", ex.getMessage(),
                    "hint", "Ensure the Python ML service is running: cd ml-service && uvicorn app.main:app --port 8001"
                )
            );
        }
    }

    /**
     * Retrieves historical Heart Disease assessments for a specific patient.
     *
     * @param patientId ID of the patient
     * @return chronological list of recorded Heart assessments
     */
    @GetMapping("/heart/patient/{patientId}")
    @Operation(
        summary = "Retrieve historical Heart disease assessments for a patient",
        description = "Retrieves all recorded Heart Disease assessments for the given patient ID in reverse chronological order."
    )
    public ResponseEntity<List<DiseaseAssessmentResponse>> getHeartAssessmentsByPatient(
            @PathVariable Long patientId) {
        List<DiseaseAssessmentResponse> responses = heartAssessmentService.getPreviousHeartAssessments(patientId);
        return ResponseEntity.ok(responses);
    }
}
