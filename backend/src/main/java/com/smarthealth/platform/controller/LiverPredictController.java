package com.smarthealth.platform.controller;

import com.smarthealth.platform.dto.LiverPredictRequest;
import com.smarthealth.platform.dto.LiverPredictResponse;
import com.smarthealth.platform.service.LiverAssessmentService;
import com.smarthealth.platform.service.MlServiceClient;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * REST controller for ML-integrated disease prediction endpoints.
 *
 * POST /api/assessments/liver/predict
 *   Accepts validated liver clinical features, calls the Python ML inference service,
 *   persists the result, and returns the full prediction + SHAP explanation to the client.
 *
 * Architecture:
 *   React → POST /api/assessments/liver/predict (Spring Boot)
 *         → POST /api/predict/liver              (Python FastAPI ML service)
 *         ← probability + SHAP explanation
 *         → persist to disease_assessments table
 *         ← LiverPredictResponse (to React)
 */
@RestController
@RequestMapping("/api/assessments")
@Tag(name = "ML Predictions", description = "ML inference endpoints integrated with the Python microservice")
public class LiverPredictController {

    private final LiverAssessmentService liverAssessmentService;

    public LiverPredictController(LiverAssessmentService liverAssessmentService) {
        this.liverAssessmentService = liverAssessmentService;
    }

    /**
     * Runs a liver disease risk prediction using the production Logistic Regression
     * model with 5-fold Isotonic Calibration. SHAP local feature contributions are
     * included in the response.
     *
     * The result is automatically persisted to the disease_assessments table.
     *
     * @param request validated liver clinical feature inputs + patientId
     * @return full prediction response (probability, SHAP, model metadata, disclaimer)
     */
    @PostMapping("/liver/predict")
    @Operation(
        summary = "Liver Disease Risk Prediction",
        description = "Runs calibrated ML inference for liver disease risk using real model outputs. " +
                      "Result is persisted and SHAP explanation is returned. No synthetic values are generated."
    )
    public ResponseEntity<?> predictLiver(@Valid @RequestBody LiverPredictRequest request) {
        try {
            LiverPredictResponse response = liverAssessmentService.predict(request);
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
}
