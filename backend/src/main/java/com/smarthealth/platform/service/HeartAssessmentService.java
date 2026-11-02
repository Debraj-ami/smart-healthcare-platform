package com.smarthealth.platform.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.smarthealth.platform.dto.DiseaseAssessmentResponse;
import com.smarthealth.platform.dto.HeartPredictRequest;
import com.smarthealth.platform.dto.HeartPredictResponse;
import com.smarthealth.platform.entity.DiseaseAssessment;
import com.smarthealth.platform.entity.DiseaseType;
import com.smarthealth.platform.entity.PersonalBaseline;
import com.smarthealth.platform.entity.User;
import com.smarthealth.platform.repository.DiseaseAssessmentRepository;
import com.smarthealth.platform.repository.PersonalBaselineRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.Period;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Orchestrates Heart Disease prediction and patient-context integration:
 *
 * 1. Validates and resolves patient if patientId is provided.
 * 2. Retrieves existing patient context (prior Heart assessments, established personal baselines).
 * 3. Forwards the 13 clinical features to the Python FastAPI ML microservice.
 * 4. Validates and normalizes the ML response (probability, threshold, prediction, SHAP).
 * 5. Persists the assessment record if patientId is present (using DiseaseType.HEART_DISEASE).
 * 6. Enriches response with verified patient context without fabricating any synthetic history.
 *
 * Responsibilities:
 * - Spring Boot: Platform orchestration, validation, persistence, patient context, and client contract.
 * - FastAPI ML Service: Preprocessing, Logistic Regression inference, threshold evaluation, and SHAP.
 */
@Service
public class HeartAssessmentService {

    private final MlServiceClient mlServiceClient;
    private final DiseaseAssessmentRepository assessmentRepository;
    private final PersonalBaselineRepository personalBaselineRepository;
    private final UserService userService;
    private final ObjectMapper objectMapper;

    public HeartAssessmentService(
            MlServiceClient mlServiceClient,
            DiseaseAssessmentRepository assessmentRepository,
            PersonalBaselineRepository personalBaselineRepository,
            UserService userService,
            ObjectMapper objectMapper) {
        this.mlServiceClient = mlServiceClient;
        this.assessmentRepository = assessmentRepository;
        this.personalBaselineRepository = personalBaselineRepository;
        this.userService = userService;
        this.objectMapper = objectMapper;
    }

    /**
     * Executes Heart disease prediction via the ML microservice.
     *
     * @param request validated 13 Heart predictor inputs (+ optional patientId)
     * @return structured prediction response with probability, threshold, and SHAP explanation
     */
    @Transactional
    public HeartPredictResponse predict(HeartPredictRequest request) {
        // 1. Optional patient resolution & context retrieval if patientId is supplied
        User patient = null;
        List<DiseaseAssessment> priorHeartAssessments = Collections.emptyList();
        List<PersonalBaseline> existingBaselines = Collections.emptyList();

        if (request.getPatientId() != null) {
            patient = userService.getUserEntity(request.getPatientId());
            priorHeartAssessments = assessmentRepository
                    .findByPatientIdAndDiseaseTypeOrderByAssessedAtDesc(patient.getId(), DiseaseType.HEART_DISEASE);
            existingBaselines = personalBaselineRepository.findByPatientId(patient.getId());
        }

        // 2. Call ML microservice
        MlServiceClient.MlHeartPredictionResponse mlResponse = mlServiceClient.predictHeart(request);

        if (mlResponse == null) {
            throw new MlServiceClient.MlServiceException("ML service returned an empty response body.");
        }

        if (mlResponse.getProbability() == null) {
            throw new MlServiceClient.MlServiceException("ML service response missing model-estimated probability.");
        }

        double probability = mlResponse.getProbability();
        Integer prediction = mlResponse.getPrediction();
        Double threshold = mlResponse.getThreshold();
        String thresholdDesc = mlResponse.getThresholdDescription();

        // 3. Assemble response DTO
        HeartPredictResponse response = new HeartPredictResponse();
        response.setDisease(mlResponse.getDisease() != null ? mlResponse.getDisease() : "HEART_DISEASE");
        response.setProbability(probability);
        response.setPrediction(prediction);
        response.setThreshold(threshold != null ? threshold : 0.31);
        response.setThresholdDescription(thresholdDesc);

        // Model metadata
        if (mlResponse.getModel() != null) {
            response.setModelName(mlResponse.getModel().getName());
            response.setModelVersion(mlResponse.getModel().getVersion());
            response.setConfigId(mlResponse.getModel().getConfigId());
            response.setOutputSpace(mlResponse.getModel().getOutputSpace());
            response.setCalibrationStatus(mlResponse.getModel().getCalibrationStatus());
        }

        // SHAP explanation mapping
        if (mlResponse.getExplanation() != null) {
            MlServiceClient.MlHeartExplanationOutput exp = mlResponse.getExplanation();
            boolean isAvailable = Boolean.TRUE.equals(exp.getAvailable());
            response.setExplanationAvailable(isAvailable);
            response.setExplanationOutputSpace(exp.getOutputSpace());
            response.setShapBaseValue(exp.getBaseValue());
            response.setModelOutput(exp.getModelOutput());
            response.setExplanationError(exp.getError());

            if (isAvailable) {
                response.setPositiveContributors(mapContributions(exp.getPositiveContributors()));
                response.setNegativeContributors(mapContributions(exp.getNegativeContributors()));
                response.setAllContributions(mapContributions(exp.getAllContributions()));
            } else {
                response.setPositiveContributors(Collections.emptyList());
                response.setNegativeContributors(Collections.emptyList());
                response.setAllContributions(Collections.emptyList());
            }
        } else {
            response.setExplanationAvailable(false);
            response.setPositiveContributors(Collections.emptyList());
            response.setNegativeContributors(Collections.emptyList());
            response.setAllContributions(Collections.emptyList());
        }

        response.setInputSummary(mlResponse.getInputSummary());
        response.setDisclaimer(mlResponse.getDisclaimer());

        // 4. Persistence & Patient Context enrichment if patient was specified
        if (patient != null) {
            // Assemble verified patient context
            HeartPredictResponse.PatientContextDto contextDto = new HeartPredictResponse.PatientContextDto();
            contextDto.setPatientId(patient.getId());
            contextDto.setGender(patient.getGender());
            if (patient.getDateOfBirth() != null) {
                contextDto.setAgeYears(Period.between(patient.getDateOfBirth(), LocalDate.now()).getYears());
            }

            int priorCount = priorHeartAssessments.size();
            contextDto.setPreviousAssessmentCount(priorCount);
            if (priorCount > 0) {
                DiseaseAssessment latestPrior = priorHeartAssessments.get(0);
                contextDto.setLastAssessedAt(latestPrior.getAssessedAt());
                contextDto.setLastProbability(latestPrior.getRiskScore());
            }

            boolean hasBaselines = !existingBaselines.isEmpty();
            contextDto.setHasEstablishedBaselines(hasBaselines);
            List<String> baselineMetrics = existingBaselines.stream()
                    .map(PersonalBaseline::getMetricType)
                    .collect(Collectors.toList());
            contextDto.setEstablishedBaselineMetrics(baselineMetrics);

            contextDto.setContextNotes(
                "Existing patient context resolved. Heart assessment inputs are screening inputs and are not automatically logged as longitudinal health observations. No synthetic baselines or trends are generated."
            );

            response.setPatientContext(contextDto);

            String shapJson = serializeShapAsJson(mlResponse, contextDto);
            DiseaseAssessment assessment = new DiseaseAssessment(
                    patient,
                    DiseaseType.HEART_DISEASE,
                    probability,
                    null, // No arbitrary categorical risk labels forced
                    shapJson
            );
            DiseaseAssessment saved = assessmentRepository.save(assessment);
            response.setAssessmentId(saved.getId());
            response.setPatientId(patient.getId());
            response.setAssessedAt(saved.getAssessedAt());
        }

        return response;
    }

    /**
     * Retrieves historical Heart Disease assessments for a specific patient.
     *
     * @param patientId verified patient identifier
     * @return chronological list of recorded Heart assessments (most recent first)
     */
    @Transactional(readOnly = true)
    public List<DiseaseAssessmentResponse> getPreviousHeartAssessments(Long patientId) {
        userService.getUserEntity(patientId);
        return assessmentRepository.findByPatientIdAndDiseaseTypeOrderByAssessedAtDesc(patientId, DiseaseType.HEART_DISEASE)
                .stream()
                .map(DiseaseAssessmentResponse::fromEntity)
                .collect(Collectors.toList());
    }

    /**
     * Maps inner ML service contribution objects to Spring Boot DTOs.
     */
    private List<HeartPredictResponse.FeatureContributionDto> mapContributions(
            List<MlServiceClient.MlHeartFeatureContribution> mlContributions) {
        if (mlContributions == null) {
            return Collections.emptyList();
        }
        List<HeartPredictResponse.FeatureContributionDto> list = new ArrayList<>();
        for (MlServiceClient.MlHeartFeatureContribution mc : mlContributions) {
            list.add(new HeartPredictResponse.FeatureContributionDto(
                    mc.getFeature(),
                    mc.getTransformedFeature(),
                    mc.getDisplayName(),
                    mc.getRawValue(),
                    mc.getContribution(),
                    mc.getAbsContribution(),
                    mc.getDirection()
            ));
        }
        return list;
    }

    /**
     * Serializes explanation and model metrics into compact JSON for the inputFeatures column.
     */
    private String serializeShapAsJson(
            MlServiceClient.MlHeartPredictionResponse mlResponse,
            HeartPredictResponse.PatientContextDto contextDto) {
        try {
            Map<String, Object> payload = new java.util.LinkedHashMap<>();
            payload.put("probability", mlResponse.getProbability());
            payload.put("prediction", mlResponse.getPrediction());
            payload.put("threshold", mlResponse.getThreshold());
            if (mlResponse.getModel() != null) {
                payload.put("modelName", mlResponse.getModel().getName());
                payload.put("modelVersion", mlResponse.getModel().getVersion());
                payload.put("configId", mlResponse.getModel().getConfigId());
                payload.put("outputSpace", mlResponse.getModel().getOutputSpace());
                payload.put("calibrationStatus", mlResponse.getModel().getCalibrationStatus());
            }
            if (contextDto != null) {
                payload.put("patientContext", contextDto);
            }
            if (mlResponse.getExplanation() != null) {
                payload.put("shapBaseValue", mlResponse.getExplanation().getBaseValue());
                payload.put("modelOutput", mlResponse.getExplanation().getModelOutput());
                payload.put("explanationAvailable", mlResponse.getExplanation().getAvailable());
                payload.put("explanationOutputSpace", mlResponse.getExplanation().getOutputSpace());
                payload.put("positiveContributors", mlResponse.getExplanation().getPositiveContributors());
                payload.put("negativeContributors", mlResponse.getExplanation().getNegativeContributors());
                payload.put("allContributions", mlResponse.getExplanation().getAllContributions());
            }
            if (mlResponse.getInputSummary() != null) {
                payload.put("inputSummary", mlResponse.getInputSummary());
            }
            return objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException e) {
            return "{\"probability\":" + mlResponse.getProbability() + "}";
        }
    }
}
