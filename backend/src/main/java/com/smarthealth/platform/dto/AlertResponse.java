package com.smarthealth.platform.dto;

import com.smarthealth.platform.entity.Alert;

import java.time.LocalDateTime;

public class AlertResponse {

    private Long id;
    private Long patientId;
    private Long assessmentId;
    private Long trendAnalysisId;
    private String alertType;
    private String severity;
    private String message;
    private String status;
    private LocalDateTime createdAt;

    public AlertResponse() {
    }

    public static AlertResponse fromEntity(Alert alert) {
        AlertResponse response = new AlertResponse();
        response.setId(alert.getId());
        response.setPatientId(alert.getPatient() != null ? alert.getPatient().getId() : null);
        response.setAssessmentId(alert.getAssessment() != null ? alert.getAssessment().getId() : null);
        response.setTrendAnalysisId(alert.getTrendAnalysis() != null ? alert.getTrendAnalysis().getId() : null);
        response.setAlertType(alert.getAlertType());
        response.setSeverity(alert.getSeverity());
        response.setMessage(alert.getMessage());
        response.setStatus(alert.getStatus());
        response.setCreatedAt(alert.getCreatedAt());
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

    public Long getAssessmentId() {
        return assessmentId;
    }

    public void setAssessmentId(Long assessmentId) {
        this.assessmentId = assessmentId;
    }

    public Long getTrendAnalysisId() {
        return trendAnalysisId;
    }

    public void setTrendAnalysisId(Long trendAnalysisId) {
        this.trendAnalysisId = trendAnalysisId;
    }

    public String getAlertType() {
        return alertType;
    }

    public void setAlertType(String alertType) {
        this.alertType = alertType;
    }

    public String getSeverity() {
        return severity;
    }

    public void setSeverity(String severity) {
        this.severity = severity;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
}
