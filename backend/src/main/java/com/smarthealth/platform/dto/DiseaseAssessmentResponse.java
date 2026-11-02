package com.smarthealth.platform.dto;

import com.smarthealth.platform.entity.DiseaseAssessment;
import com.smarthealth.platform.entity.DiseaseType;
import com.smarthealth.platform.entity.RiskLevel;

import java.time.LocalDateTime;

public class DiseaseAssessmentResponse {

    private Long id;
    private Long patientId;
    private DiseaseType diseaseType;
    private Double riskScore;
    private RiskLevel riskLevel;
    private String inputFeatures;
    private LocalDateTime assessedAt;

    public DiseaseAssessmentResponse() {
    }

    public static DiseaseAssessmentResponse fromEntity(DiseaseAssessment assessment) {
        DiseaseAssessmentResponse response = new DiseaseAssessmentResponse();
        response.setId(assessment.getId());
        response.setPatientId(assessment.getPatient() != null ? assessment.getPatient().getId() : null);
        response.setDiseaseType(assessment.getDiseaseType());
        response.setRiskScore(assessment.getRiskScore());
        response.setRiskLevel(assessment.getRiskLevel());
        response.setInputFeatures(assessment.getInputFeatures());
        response.setAssessedAt(assessment.getAssessedAt());
        return response;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
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

    public LocalDateTime getAssessedAt() {
        return assessedAt;
    }

    public void setAssessedAt(LocalDateTime assessedAt) {
        this.assessedAt = assessedAt;
    }
}
