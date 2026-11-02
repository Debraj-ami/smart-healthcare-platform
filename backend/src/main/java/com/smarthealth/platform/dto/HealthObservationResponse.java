package com.smarthealth.platform.dto;

import com.smarthealth.platform.entity.HealthObservation;

import java.time.LocalDateTime;

public class HealthObservationResponse {

    private Long id;
    private Long patientId;
    private String metricType;
    private Double metricValue;
    private String unit;
    private LocalDateTime loggedAt;
    private com.smarthealth.platform.entity.ActivityLevel activityLevel;
    private Integer activityMinutes;

    public HealthObservationResponse() {
    }

    public static HealthObservationResponse fromEntity(HealthObservation observation) {
        HealthObservationResponse response = new HealthObservationResponse();
        response.setId(observation.getId());
        response.setPatientId(observation.getPatient() != null ? observation.getPatient().getId() : null);
        response.setMetricType(observation.getMetricType());
        response.setMetricValue(observation.getMetricValue());
        response.setUnit(observation.getUnit());
        response.setLoggedAt(observation.getLoggedAt());
        response.setActivityLevel(observation.getActivityLevel());
        response.setActivityMinutes(observation.getActivityMinutes());
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

    public String getMetricType() {
        return metricType;
    }

    public void setMetricType(String metricType) {
        this.metricType = metricType;
    }

    public Double getMetricValue() {
        return metricValue;
    }

    public void setMetricValue(Double metricValue) {
        this.metricValue = metricValue;
    }

    public String getUnit() {
        return unit;
    }

    public void setUnit(String unit) {
        this.unit = unit;
    }

    public LocalDateTime getLoggedAt() {
        return loggedAt;
    }

    public void setLoggedAt(LocalDateTime loggedAt) {
        this.loggedAt = loggedAt;
    }

    public com.smarthealth.platform.entity.ActivityLevel getActivityLevel() {
        return activityLevel;
    }

    public void setActivityLevel(com.smarthealth.platform.entity.ActivityLevel activityLevel) {
        this.activityLevel = activityLevel;
    }

    public Integer getActivityMinutes() {
        return activityMinutes;
    }

    public void setActivityMinutes(Integer activityMinutes) {
        this.activityMinutes = activityMinutes;
    }
}
