package com.smarthealth.platform.service;

import com.smarthealth.platform.dto.LiverPredictRequest;
import com.smarthealth.platform.dto.LiverPredictResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;

import java.util.*;

/**
 * HTTP client that Spring Boot uses to call the Python FastAPI ML microservice.
 *
 * Responsibility boundary:
 *   - Spring Boot (this class): request marshalling, error translation, retry/timeout handling
 *   - Python ML service: preprocessing, model inference, calibration, SHAP explanation
 *
 * The base URL is configured via application.yml (ml.service.base-url) and can be
 * overridden per environment with the ML_SERVICE_BASE_URL environment variable.
 */
@Service
public class MlServiceClient {

    private final RestTemplate restTemplate;
    private final String mlServiceBaseUrl;

    public MlServiceClient(
            RestTemplate restTemplate,
            @Value("${ml.service.base-url}") String mlServiceBaseUrl) {
        this.restTemplate = restTemplate;
        this.mlServiceBaseUrl = mlServiceBaseUrl;
    }

    /**
     * Calls POST /api/predict/liver on the Python ML service.
     *
     * @param request the validated liver disease feature inputs
     * @return the raw prediction response from the ML service (probability + SHAP)
     * @throws MlServiceException if the ML service is unavailable or returns an error
     */
    public MlLiverPredictionResponse predictLiver(LiverPredictRequest request) {
        String url = mlServiceBaseUrl + "/api/predict/liver";

        // Build ML service payload — maps Spring DTO fields to Python schema names
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("age", request.getAge());
        payload.put("gender", request.getGender());
        payload.put("totalBilirubin", request.getTotalBilirubin());
        payload.put("directBilirubin", request.getDirectBilirubin());
        payload.put("alkphos", request.getAlkalinePhosphatase());
        payload.put("alkalinePhosphatase", request.getAlkalinePhosphatase());
        payload.put("sgpt", request.getSgpt());
        payload.put("sgot", request.getSgot());
        payload.put("totalProteins", request.getTotalProteins());
        payload.put("albumin", request.getAlbumin());
        payload.put("agRatio", request.getAgRatio());

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(payload, headers);

        try {
            ResponseEntity<MlLiverPredictionResponse> response =
                    restTemplate.exchange(url, HttpMethod.POST, entity, MlLiverPredictionResponse.class);

            if (response.getBody() == null) {
                throw new MlServiceException("ML service returned an empty response body.");
            }
            return response.getBody();

        } catch (ResourceAccessException ex) {
            throw new MlServiceException(
                    "ML service is unavailable at " + mlServiceBaseUrl + ". " +
                    "Ensure the Python service is running (uvicorn app.main:app --port 8001). " +
                    "Cause: " + ex.getMessage(), ex);
        } catch (HttpClientErrorException ex) {
            throw new MlServiceException(
                    "ML service rejected the request (HTTP " + ex.getStatusCode() + "): " + ex.getResponseBodyAsString(), ex);
        } catch (HttpServerErrorException ex) {
            throw new MlServiceException(
                    "ML service internal error (HTTP " + ex.getStatusCode() + "): " + ex.getResponseBodyAsString(), ex);
        }
    }

    // =====================================================================
    // Inner DTOs that mirror the Python LiverPredictionResponse schema
    // These are only used internally by the client for JSON deserialization.
    // =====================================================================

    public static class MlLiverPredictionResponse {
        private String disease;
        private MlModelInfo model;
        private Double probability;
        private MlExplanationOutput explanation;
        private Map<String, Object> inputSummary;
        private String disclaimer;

        public String getDisease() { return disease; }
        public void setDisease(String disease) { this.disease = disease; }

        public MlModelInfo getModel() { return model; }
        public void setModel(MlModelInfo model) { this.model = model; }

        public Double getProbability() { return probability; }
        public void setProbability(Double probability) { this.probability = probability; }

        public MlExplanationOutput getExplanation() { return explanation; }
        public void setExplanation(MlExplanationOutput explanation) { this.explanation = explanation; }

        public Map<String, Object> getInputSummary() { return inputSummary; }
        public void setInputSummary(Map<String, Object> inputSummary) { this.inputSummary = inputSummary; }

        public String getDisclaimer() { return disclaimer; }
        public void setDisclaimer(String disclaimer) { this.disclaimer = disclaimer; }
    }

    public static class MlModelInfo {
        private String name;
        private String version;
        private String calibrationMethod;

        public String getName() { return name; }
        public void setName(String name) { this.name = name; }

        public String getVersion() { return version; }
        public void setVersion(String version) { this.version = version; }

        public String getCalibrationMethod() { return calibrationMethod; }
        public void setCalibrationMethod(String calibrationMethod) { this.calibrationMethod = calibrationMethod; }
    }

    public static class MlExplanationOutput {
        private Double baseValue;
        private String outputSpace;
        private List<MlFeatureContribution> features;

        public Double getBaseValue() { return baseValue; }
        public void setBaseValue(Double baseValue) { this.baseValue = baseValue; }

        public String getOutputSpace() { return outputSpace; }
        public void setOutputSpace(String outputSpace) { this.outputSpace = outputSpace; }

        public List<MlFeatureContribution> getFeatures() { return features; }
        public void setFeatures(List<MlFeatureContribution> features) { this.features = features; }
    }

    public static class MlFeatureContribution {
        private String feature;
        private String transformedFeature;
        private Object value;
        private Double contribution;
        private String direction;

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

    // =====================================================================
    // Heart Disease Prediction (Productionization Stage 2)
    // =====================================================================

    /**
     * Calls POST /api/predict/heart on the Python ML service.
     *
     * @param request the validated heart disease feature inputs
     * @return the raw prediction response from the ML service (probability, prediction, threshold, SHAP)
     * @throws MlServiceException if the ML service is unavailable or returns an error
     */
    public MlHeartPredictionResponse predictHeart(com.smarthealth.platform.dto.HeartPredictRequest request) {
        String url = mlServiceBaseUrl + "/api/predict/heart";

        // Build ML service payload — maps Spring DTO fields to Python schema names
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("age", request.getAge());
        payload.put("sex", request.getSex());
        payload.put("cp", request.getCp());
        payload.put("trestbps", request.getTrestbps());
        payload.put("chol", request.getChol());
        payload.put("fbs", request.getFbs());
        payload.put("restecg", request.getRestecg());
        payload.put("thalach", request.getThalach());
        payload.put("exang", request.getExang());
        payload.put("oldpeak", request.getOldpeak());
        payload.put("slope", request.getSlope());
        payload.put("ca", request.getCa());
        payload.put("thal", request.getThal());

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(payload, headers);

        try {
            ResponseEntity<MlHeartPredictionResponse> response =
                    restTemplate.exchange(url, HttpMethod.POST, entity, MlHeartPredictionResponse.class);

            if (response.getBody() == null) {
                throw new MlServiceException("ML service returned an empty response body.");
            }
            return response.getBody();

        } catch (ResourceAccessException ex) {
            throw new MlServiceException(
                    "ML service is unavailable at " + mlServiceBaseUrl + ". " +
                    "Ensure the Python service is running (cd ml-service && uvicorn app.main:app --port 8001). " +
                    "Cause: " + ex.getMessage(), ex);
        } catch (org.springframework.web.client.RestClientResponseException ex) {
            throw new MlServiceException(
                    "ML service returned an error (HTTP " + ex.getStatusCode() + "): " + ex.getResponseBodyAsString(), ex);
        } catch (org.springframework.web.client.RestClientException ex) {
            throw new MlServiceException(
                    "ML service request failed: " + ex.getMessage(), ex);
        }
    }

    @com.fasterxml.jackson.annotation.JsonIgnoreProperties(ignoreUnknown = true)
    public static class MlHeartPredictionResponse {
        private String disease;
        private MlHeartModelInfo model;
        private Double probability;
        private Integer prediction;
        private Double threshold;
        @com.fasterxml.jackson.annotation.JsonAlias({"threshold_description", "thresholdDescription"})
        private String thresholdDescription;
        private MlHeartExplanationOutput explanation;
        @com.fasterxml.jackson.annotation.JsonAlias({"input_summary", "inputSummary"})
        private Map<String, Object> inputSummary;
        private String disclaimer;

        public String getDisease() { return disease; }
        public void setDisease(String disease) { this.disease = disease; }

        public MlHeartModelInfo getModel() { return model; }
        public void setModel(MlHeartModelInfo model) { this.model = model; }

        public Double getProbability() { return probability; }
        public void setProbability(Double probability) { this.probability = probability; }

        public Integer getPrediction() { return prediction; }
        public void setPrediction(Integer prediction) { this.prediction = prediction; }

        public Double getThreshold() { return threshold; }
        public void setThreshold(Double threshold) { this.threshold = threshold; }

        public String getThresholdDescription() { return thresholdDescription; }
        public void setThresholdDescription(String thresholdDescription) { this.thresholdDescription = thresholdDescription; }

        public MlHeartExplanationOutput getExplanation() { return explanation; }
        public void setExplanation(MlHeartExplanationOutput explanation) { this.explanation = explanation; }

        public Map<String, Object> getInputSummary() { return inputSummary; }
        public void setInputSummary(Map<String, Object> inputSummary) { this.inputSummary = inputSummary; }

        public String getDisclaimer() { return disclaimer; }
        public void setDisclaimer(String disclaimer) { this.disclaimer = disclaimer; }
    }

    @com.fasterxml.jackson.annotation.JsonIgnoreProperties(ignoreUnknown = true)
    public static class MlHeartModelInfo {
        private String name;
        private String version;
        @com.fasterxml.jackson.annotation.JsonAlias({"config_id", "configId"})
        private String configId;
        @com.fasterxml.jackson.annotation.JsonAlias({"output_space", "outputSpace"})
        private String outputSpace;
        @com.fasterxml.jackson.annotation.JsonAlias({"calibration_status", "calibrationStatus"})
        private String calibrationStatus;

        public String getName() { return name; }
        public void setName(String name) { this.name = name; }

        public String getVersion() { return version; }
        public void setVersion(String version) { this.version = version; }

        public String getConfigId() { return configId; }
        public void setConfigId(String configId) { this.configId = configId; }

        public String getOutputSpace() { return outputSpace; }
        public void setOutputSpace(String outputSpace) { this.outputSpace = outputSpace; }

        public String getCalibrationStatus() { return calibrationStatus; }
        public void setCalibrationStatus(String calibrationStatus) { this.calibrationStatus = calibrationStatus; }
    }

    @com.fasterxml.jackson.annotation.JsonIgnoreProperties(ignoreUnknown = true)
    public static class MlHeartExplanationOutput {
        private Boolean available;
        @com.fasterxml.jackson.annotation.JsonAlias({"output_space", "outputSpace"})
        private String outputSpace;
        @com.fasterxml.jackson.annotation.JsonAlias({"base_value", "baseValue"})
        private Double baseValue;
        @com.fasterxml.jackson.annotation.JsonAlias({"model_output", "modelOutput"})
        private Double modelOutput;
        @com.fasterxml.jackson.annotation.JsonAlias({"positive_contributors", "positiveContributors"})
        private List<MlHeartFeatureContribution> positiveContributors;
        @com.fasterxml.jackson.annotation.JsonAlias({"negative_contributors", "negativeContributors"})
        private List<MlHeartFeatureContribution> negativeContributors;
        @com.fasterxml.jackson.annotation.JsonAlias({"all_contributions", "allContributions"})
        private List<MlHeartFeatureContribution> allContributions;
        private String error;

        public Boolean getAvailable() { return available; }
        public void setAvailable(Boolean available) { this.available = available; }

        public String getOutputSpace() { return outputSpace; }
        public void setOutputSpace(String outputSpace) { this.outputSpace = outputSpace; }

        public Double getBaseValue() { return baseValue; }
        public void setBaseValue(Double baseValue) { this.baseValue = baseValue; }

        public Double getModelOutput() { return modelOutput; }
        public void setModelOutput(Double modelOutput) { this.modelOutput = modelOutput; }

        public List<MlHeartFeatureContribution> getPositiveContributors() { return positiveContributors; }
        public void setPositiveContributors(List<MlHeartFeatureContribution> positiveContributors) { this.positiveContributors = positiveContributors; }

        public List<MlHeartFeatureContribution> getNegativeContributors() { return negativeContributors; }
        public void setNegativeContributors(List<MlHeartFeatureContribution> negativeContributors) { this.negativeContributors = negativeContributors; }

        public List<MlHeartFeatureContribution> getAllContributions() { return allContributions; }
        public void setAllContributions(List<MlHeartFeatureContribution> allContributions) { this.allContributions = allContributions; }

        public String getError() { return error; }
        public void setError(String error) { this.error = error; }
    }

    @com.fasterxml.jackson.annotation.JsonIgnoreProperties(ignoreUnknown = true)
    public static class MlHeartFeatureContribution {
        private String feature;
        @com.fasterxml.jackson.annotation.JsonAlias({"transformed_feature", "transformedFeature"})
        private String transformedFeature;
        @com.fasterxml.jackson.annotation.JsonAlias({"display_name", "displayName"})
        private String displayName;
        @com.fasterxml.jackson.annotation.JsonAlias({"raw_value", "rawValue"})
        private Object rawValue;
        private Double contribution;
        @com.fasterxml.jackson.annotation.JsonAlias({"abs_contribution", "absContribution"})
        private Double absContribution;
        private String direction;

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

    /**
     * Checked exception representing a communication or protocol error with the ML service.
     * Callers should map this to an appropriate HTTP error response.
     */
    public static class MlServiceException extends RuntimeException {
        public MlServiceException(String message) { super(message); }
        public MlServiceException(String message, Throwable cause) { super(message, cause); }
    }
}

