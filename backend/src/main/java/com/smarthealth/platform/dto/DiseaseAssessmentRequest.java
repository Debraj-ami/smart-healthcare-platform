package com.smarthealth.platform.dto;

import com.smarthealth.platform.entity.DiseaseType;
import com.smarthealth.platform.entity.RiskLevel;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotNull;

public class DiseaseAssessmentRequest {

    @NotNull(message = "Patient ID is required")
    private Long patientId;

    @NotNull(message = "Disease type is required")
    private DiseaseType diseaseType;

    @NotNull(message = "Risk score is required")
    @DecimalMin(value = "0.0", message = "Risk score must be at least 0.0")
    @DecimalMax(value = "1.0", message = "Risk score must be at most 1.0")
    private Double riskScore;

    @NotNull(message = "Risk level is required")
    private RiskLevel riskLevel;

    private String inputFeatures;

    public DiseaseAssessmentRequest() {
    }

    public DiseaseAssessmentRequest(Long patientId, DiseaseType diseaseType, Double riskScore,
                                    RiskLevel riskLevel, String inputFeatures) {
        this.patientId = patientId;
        this.diseaseType = diseaseType;
        this.riskScore = riskScore;
        this.riskLevel = riskLevel;
        this.inputFeatures = inputFeatures;
    }

    public Long getPatientId() {
        return patientId;
    }

    public void setPatientId(Long patientId) {
        this.patientId = patientId;
    }

    public DiseaseType getDiseaseType() {
        return diseaseType;
    }

    public void setDiseaseType(DiseaseType diseaseType) {
        this.diseaseType = diseaseType;
    }

    public Double getRiskScore() {
        return riskScore;
    }

    public void setRiskScore(Double riskScore) {
        this.riskScore = riskScore;
    }

    public RiskLevel getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(RiskLevel riskLevel) {
        this.riskLevel = riskLevel;
    }

    public String getInputFeatures() {
        return inputFeatures;
    }

    public void setInputFeatures(String inputFeatures) {
        this.inputFeatures = inputFeatures;
    }
}
