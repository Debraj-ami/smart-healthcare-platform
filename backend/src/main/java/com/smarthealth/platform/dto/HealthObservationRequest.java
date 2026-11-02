package com.smarthealth.platform.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

public class HealthObservationRequest {

    @NotNull(message = "Patient ID is required")
    private Long patientId;

    @NotBlank(message = "Metric type is required")
    private String metricType;

    @NotNull(message = "Metric value is required")
    private Double metricValue;

    private String unit;

    private com.smarthealth.platform.entity.ActivityLevel activityLevel;

    private Integer activityMinutes;

    public HealthObservationRequest() {
    }

    public HealthObservationRequest(Long patientId, String metricType, Double metricValue, String unit) {
        this.patientId = patientId;
        this.metricType = metricType;
        this.metricValue = metricValue;
        this.unit = unit;
        this.activityLevel = com.smarthealth.platform.entity.ActivityLevel.UNKNOWN;
    }

    public HealthObservationRequest(Long patientId, String metricType, Double metricValue, String unit,
                                  com.smarthealth.platform.entity.ActivityLevel activityLevel, Integer activityMinutes) {
        this.patientId = patientId;
        this.metricType = metricType;
        this.metricValue = metricValue;
        this.unit = unit;
        this.activityLevel = activityLevel != null ? activityLevel : com.smarthealth.platform.entity.ActivityLevel.UNKNOWN;
        this.activityMinutes = activityMinutes;
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
