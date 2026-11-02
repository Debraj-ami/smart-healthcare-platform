package com.smarthealth.platform.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * Response returned by POST /api/assessments/liver/predict
 *
 * Contains the ML prediction result from the Python service, the persisted
 * assessment record ID, and the full SHAP explanation — enabling the frontend
 * to display probability, feature contributions, and the research disclaimer.
 *
 * No synthetic or hardcoded predictions are included. Every field is populated
 * from real model inference at request time.
 */
public class LiverPredictResponse {

    // Assessment record persisted in disease_assessments table
    private Long assessmentId;
    private Long patientId;
    private LocalDateTime assessedAt;

    // Core ML output
    private String disease;
    private Double probability;

    // Model metadata
    private String modelName;
    private String modelVersion;
    private String calibrationMethod;

    // SHAP explanation
    private Double shapBaseValue;
    private String shapOutputSpace;
    private List<FeatureContributionDto> shapFeatures;

    // Original submitted values echoed back
    private Map<String, Object> inputSummary;

    // Mandatory research disclaimer
    private String disclaimer;

    public LiverPredictResponse() {}

    // --- Inner DTO for SHAP feature contributions ---
    public static class FeatureContributionDto {
        private String feature;
        private String transformedFeature;
        private Object value;
        private Double contribution;
        private String direction;

        public FeatureContributionDto() {}

        public String getFeature() { return feature; }
        public void setFeature(String feature) { this.feature = feature; }

        public String getTransformedFeature() { return transformedFeature; }
        public void setTransformedFeature(String transformedFeature) { this.transformedFeature = transformedFeature; }

        public Object getValue() { return value; }
        public void setValue(Object value) { this.value = value; }

        public Double getContribution() { return contribution; }
        public void setContribution(Double contribution) { this.contribution = contribution; }

        public String getDirection() { return direction; }
        public void setDirection(String direction) { this.direction = direction; }
    }

    // --- Getters and Setters ---

    public Long getAssessmentId() { return assessmentId; }
    public void setAssessmentId(Long assessmentId) { this.assessmentId = assessmentId; }

    public Long getPatientId() { return patientId; }
    public void setPatientId(Long patientId) { this.patientId = patientId; }

    public LocalDateTime getAssessedAt() { return assessedAt; }
    public void setAssessedAt(LocalDateTime assessedAt) { this.assessedAt = assessedAt; }

    public String getDisease() { return disease; }
    public void setDisease(String disease) { this.disease = disease; }

    public Double getProbability() { return probability; }
    public void setProbability(Double probability) { this.probability = probability; }

    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }

    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }

    public String getCalibrationMethod() { return calibrationMethod; }
    public void setCalibrationMethod(String calibrationMethod) { this.calibrationMethod = calibrationMethod; }

    public Double getShapBaseValue() { return shapBaseValue; }
    public void setShapBaseValue(Double shapBaseValue) { this.shapBaseValue = shapBaseValue; }

    public String getShapOutputSpace() { return shapOutputSpace; }
    public void setShapOutputSpace(String shapOutputSpace) { this.shapOutputSpace = shapOutputSpace; }

    public List<FeatureContributionDto> getShapFeatures() { return shapFeatures; }
    public void setShapFeatures(List<FeatureContributionDto> shapFeatures) { this.shapFeatures = shapFeatures; }

    public Map<String, Object> getInputSummary() { return inputSummary; }
    public void setInputSummary(Map<String, Object> inputSummary) { this.inputSummary = inputSummary; }

    public String getDisclaimer() { return disclaimer; }
    public void setDisclaimer(String disclaimer) { this.disclaimer = disclaimer; }
}
