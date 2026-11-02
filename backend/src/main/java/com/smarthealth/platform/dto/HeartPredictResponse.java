package com.smarthealth.platform.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * Response returned by POST /api/assessments/heart/predict
 *
 * Exposes the full Heart Disease ML inference results from the Python microservice,
 * including model-estimated probability, binary classification decision against the
 * configured threshold (t=0.31), and machine-readable SHAP explanations in log-odds space.
 *
 * No synthetic or fabricated values are ever generated.
 */
public class HeartPredictResponse {

    // Persistence metadata (populated if patientId was provided in request)
    private Long assessmentId;
    private Long patientId;
    private LocalDateTime assessedAt;

    // Patient Context & Personalization integration (populated when patientId was provided)
    private PatientContextDto patientContext;

    // Core ML output
    private String disease;
    private Double probability;
    private Integer prediction;
    private Double threshold;
    private String thresholdDescription;

    // Model metadata
    private String modelName;
    private String modelVersion;
    private String configId;
    private String outputSpace;
    private String calibrationStatus;

    // SHAP explanation metadata
    private Boolean explanationAvailable;
    private String explanationOutputSpace;
    private Double shapBaseValue;
    private Double modelOutput;
    private List<FeatureContributionDto> positiveContributors;
    private List<FeatureContributionDto> negativeContributors;
    private List<FeatureContributionDto> allContributions;
    private String explanationError;

    // Echoed clinical inputs
    private Map<String, Object> inputSummary;

    // Mandatory research disclaimer
    private String disclaimer;

    public HeartPredictResponse() {}

    /**
     * Inner DTO representing a directional feature contribution in log-odds space.
     */
    public static class FeatureContributionDto {
        private String feature;
        private String transformedFeature;
        private String displayName;
        private Object rawValue;
        private Double contribution;
        private Double absContribution;
        private String direction;

        public FeatureContributionDto() {}

        public FeatureContributionDto(String feature, String transformedFeature, String displayName,
                                      Object rawValue, Double contribution, Double absContribution,
                                      String direction) {
            this.feature = feature;
            this.transformedFeature = transformedFeature;
            this.displayName = displayName;
            this.rawValue = rawValue;
            this.contribution = contribution;
            this.absContribution = absContribution;
            this.direction = direction;
        }

        public String getFeature() { return feature; }
        public void setFeature(String feature) { this.feature = feature; }

        public String getTransformedFeature() { return transformedFeature; }
        public void setTransformedFeature(String transformedFeature) { this.transformedFeature = transformedFeature; }

        public String getDisplayName() { return displayName; }
        public void setDisplayName(String displayName) { this.displayName = displayName; }

        public Object getRawValue() { return rawValue; }
        public void setRawValue(Object rawValue) { this.rawValue = rawValue; }

        public Double getContribution() { return contribution; }
        public void setContribution(Double contribution) { this.contribution = contribution; }

        public Double getAbsContribution() { return absContribution; }
        public void setAbsContribution(Double absContribution) { this.absContribution = absContribution; }

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

    public Integer getPrediction() { return prediction; }
    public void setPrediction(Integer prediction) { this.prediction = prediction; }

    public Double getThreshold() { return threshold; }
    public void setThreshold(Double threshold) { this.threshold = threshold; }

    public String getThresholdDescription() { return thresholdDescription; }
    public void setThresholdDescription(String thresholdDescription) { this.thresholdDescription = thresholdDescription; }

    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }

    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }

    public String getConfigId() { return configId; }
    public void setConfigId(String configId) { this.configId = configId; }

    public String getOutputSpace() { return outputSpace; }
    public void setOutputSpace(String outputSpace) { this.outputSpace = outputSpace; }

    public String getCalibrationStatus() { return calibrationStatus; }
    public void setCalibrationStatus(String calibrationStatus) { this.calibrationStatus = calibrationStatus; }

    public Boolean getExplanationAvailable() { return explanationAvailable; }
    public void setExplanationAvailable(Boolean explanationAvailable) { this.explanationAvailable = explanationAvailable; }

    public String getExplanationOutputSpace() { return explanationOutputSpace; }
    public void setExplanationOutputSpace(String explanationOutputSpace) { this.explanationOutputSpace = explanationOutputSpace; }

    public Double getShapBaseValue() { return shapBaseValue; }
    public void setShapBaseValue(Double shapBaseValue) { this.shapBaseValue = shapBaseValue; }

    public Double getModelOutput() { return modelOutput; }
    public void setModelOutput(Double modelOutput) { this.modelOutput = modelOutput; }

    public List<FeatureContributionDto> getPositiveContributors() { return positiveContributors; }
    public void setPositiveContributors(List<FeatureContributionDto> positiveContributors) { this.positiveContributors = positiveContributors; }

    public List<FeatureContributionDto> getNegativeContributors() { return negativeContributors; }
    public void setNegativeContributors(List<FeatureContributionDto> negativeContributors) { this.negativeContributors = negativeContributors; }

    public List<FeatureContributionDto> getAllContributions() { return allContributions; }
    public void setAllContributions(List<FeatureContributionDto> allContributions) { this.allContributions = allContributions; }

    public String getExplanationError() { return explanationError; }
    public void setExplanationError(String explanationError) { this.explanationError = explanationError; }

    public Map<String, Object> getInputSummary() { return inputSummary; }
    public void setInputSummary(Map<String, Object> inputSummary) { this.inputSummary = inputSummary; }

    public PatientContextDto getPatientContext() { return patientContext; }
    public void setPatientContext(PatientContextDto patientContext) { this.patientContext = patientContext; }

    public String getDisclaimer() { return disclaimer; }
    public void setDisclaimer(String disclaimer) { this.disclaimer = disclaimer; }

    /**
     * Inner DTO representing existing patient context and personalization state.
     * Contains only verified platform records. No synthetic history, baseline, or trend is fabricated.
     */
    public static class PatientContextDto {
        private Long patientId;
        private String gender;
        private Integer ageYears;
        private Integer previousAssessmentCount;
        private LocalDateTime lastAssessedAt;
        private Double lastProbability;
        private Boolean hasEstablishedBaselines;
        private List<String> establishedBaselineMetrics;
        private String contextNotes;

        public PatientContextDto() {}

        public PatientContextDto(Long patientId, String gender, Integer ageYears,
                                 Integer previousAssessmentCount, LocalDateTime lastAssessedAt,
                                 Double lastProbability, Boolean hasEstablishedBaselines,
                                 List<String> establishedBaselineMetrics, String contextNotes) {
            this.patientId = patientId;
            this.gender = gender;
            this.ageYears = ageYears;
            this.previousAssessmentCount = previousAssessmentCount;
            this.lastAssessedAt = lastAssessedAt;
            this.lastProbability = lastProbability;
            this.hasEstablishedBaselines = hasEstablishedBaselines;
            this.establishedBaselineMetrics = establishedBaselineMetrics;
            this.contextNotes = contextNotes;
        }

        public Long getPatientId() { return patientId; }
        public void setPatientId(Long patientId) { this.patientId = patientId; }

        public String getGender() { return gender; }
        public void setGender(String gender) { this.gender = gender; }

        public Integer getAgeYears() { return ageYears; }
        public void setAgeYears(Integer ageYears) { this.ageYears = ageYears; }

        public Integer getPreviousAssessmentCount() { return previousAssessmentCount; }
        public void setPreviousAssessmentCount(Integer previousAssessmentCount) { this.previousAssessmentCount = previousAssessmentCount; }

        public LocalDateTime getLastAssessedAt() { return lastAssessedAt; }
        public void setLastAssessedAt(LocalDateTime lastAssessedAt) { this.lastAssessedAt = lastAssessedAt; }

        public Double getLastProbability() { return lastProbability; }
        public void setLastProbability(Double lastProbability) { this.lastProbability = lastProbability; }

        public Boolean getHasEstablishedBaselines() { return hasEstablishedBaselines; }
        public void setHasEstablishedBaselines(Boolean hasEstablishedBaselines) { this.hasEstablishedBaselines = hasEstablishedBaselines; }

        public List<String> getEstablishedBaselineMetrics() { return establishedBaselineMetrics; }
        public void setEstablishedBaselineMetrics(List<String> establishedBaselineMetrics) { this.establishedBaselineMetrics = establishedBaselineMetrics; }

        public String getContextNotes() { return contextNotes; }
        public void setContextNotes(String contextNotes) { this.contextNotes = contextNotes; }
    }
}
