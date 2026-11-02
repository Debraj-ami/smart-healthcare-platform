package com.smarthealth.platform.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.smarthealth.platform.dto.LiverPredictRequest;
import com.smarthealth.platform.dto.LiverPredictResponse;
import com.smarthealth.platform.entity.DiseaseAssessment;
import com.smarthealth.platform.entity.DiseaseType;
import com.smarthealth.platform.entity.User;
import com.smarthealth.platform.repository.DiseaseAssessmentRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Orchestrates the liver disease prediction flow:
 *
 *   1. Validate patient existence
 *   2. Forward clinical inputs to the Python ML service for inference
 *   3. Persist the assessment record (calibrated probability as riskScore, SHAP as JSON, riskLevel as null)
 *   4. Assemble and return the full LiverPredictResponse to the controller
 *
 * No synthetic or hardcoded predictions are generated here.
 * The Python ML service owns all preprocessing, model inference, calibration,
 * and SHAP explanation. Spring Boot owns orchestration and persistence.
 */
@Service
public class LiverAssessmentService {

    private final MlServiceClient mlServiceClient;
    private final DiseaseAssessmentRepository assessmentRepository;
    private final UserService userService;
    private final ObjectMapper objectMapper;

    public LiverAssessmentService(
            MlServiceClient mlServiceClient,
            DiseaseAssessmentRepository assessmentRepository,
            UserService userService,
            ObjectMapper objectMapper) {
        this.mlServiceClient = mlServiceClient;
        this.assessmentRepository = assessmentRepository;
        this.userService = userService;
        this.objectMapper = objectMapper;
    }

    /**
     * Runs a liver disease prediction and persists the result.
     *
     * @param request validated liver feature inputs plus patientId
     * @return full response including ML probability, SHAP, and persisted assessment metadata
     */
    @Transactional
    public LiverPredictResponse predict(LiverPredictRequest request) {
        // 1. Verify patient exists
        User patient = userService.getUserEntity(request.getPatientId());

        // 2. Call ML service — may throw MlServiceClient.MlServiceException
        MlServiceClient.MlLiverPredictionResponse mlResponse = mlServiceClient.predictLiver(request);

        double probability = mlResponse.getProbability() != null ? mlResponse.getProbability() : 0.0;

        // 3. Serialize SHAP explanation as JSON string for inputFeatures column
        String shapJson = serializeShapAsJson(mlResponse);

        // 4. Persist the assessment record.
        //    Liver Disease risk categories (<0.35 LOW, 0.35-0.65 MODERATE, >=0.65 HIGH)
        //    were not established by research and must not be forced. We store the
        //    genuine calibrated probability in riskScore with riskLevel as null.
        DiseaseAssessment assessment = new DiseaseAssessment(
                patient,
                DiseaseType.LIVER_DISEASE,
                probability,
                null,
                shapJson
        );
        DiseaseAssessment saved = assessmentRepository.save(assessment);

        // 5. Assemble the response
        LiverPredictResponse response = new LiverPredictResponse();
        response.setAssessmentId(saved.getId());
        response.setPatientId(patient.getId());
        response.setAssessedAt(saved.getAssessedAt());
        response.setDisease(mlResponse.getDisease());
        response.setProbability(probability);

        if (mlResponse.getModel() != null) {
            response.setModelName(mlResponse.getModel().getName());
            response.setModelVersion(mlResponse.getModel().getVersion());
            response.setCalibrationMethod(mlResponse.getModel().getCalibrationMethod());
        }

        if (mlResponse.getExplanation() != null) {
            response.setShapBaseValue(mlResponse.getExplanation().getBaseValue());
            response.setShapOutputSpace(mlResponse.getExplanation().getOutputSpace());
            response.setShapFeatures(mapShapFeatures(mlResponse.getExplanation().getFeatures()));
        }

        response.setInputSummary(mlResponse.getInputSummary());
        response.setDisclaimer(mlResponse.getDisclaimer());

        return response;
    }

    /**
     * Serializes the SHAP explanation from the ML response into a compact JSON string
     * suitable for storage in the inputFeatures TEXT column.
     */
    private String serializeShapAsJson(MlServiceClient.MlLiverPredictionResponse mlResponse) {
        try {
            Map<String, Object> shapPayload = new java.util.LinkedHashMap<>();
            shapPayload.put("probability", mlResponse.getProbability());
            if (mlResponse.getExplanation() != null) {
                shapPayload.put("shapBaseValue", mlResponse.getExplanation().getBaseValue());
                shapPayload.put("shapFeatures", mlResponse.getExplanation().getFeatures());
            }
            if (mlResponse.getInputSummary() != null) {
                shapPayload.put("inputSummary", mlResponse.getInputSummary());
            }
            return objectMapper.writeValueAsString(shapPayload);
        } catch (JsonProcessingException e) {
            // Fallback: store probability only
            return "{\"probability\":" + mlResponse.getProbability() + "}";
        }
    }

    /**
     * Maps ML service inner FeatureContribution objects to the Spring DTO equivalent.
     */
    private List<LiverPredictResponse.FeatureContributionDto> mapShapFeatures(
            List<MlServiceClient.MlFeatureContribution> mlFeatures) {
        if (mlFeatures == null) return new ArrayList<>();
        List<LiverPredictResponse.FeatureContributionDto> result = new ArrayList<>();
        for (MlServiceClient.MlFeatureContribution mf : mlFeatures) {
            LiverPredictResponse.FeatureContributionDto dto = new LiverPredictResponse.FeatureContributionDto();
            dto.setFeature(mf.getFeature());
            dto.setTransformedFeature(mf.getTransformedFeature());
            dto.setValue(mf.getValue());
            dto.setContribution(mf.getContribution());
            dto.setDirection(mf.getDirection());
            result.add(dto);
        }
        return result;
    }
}
