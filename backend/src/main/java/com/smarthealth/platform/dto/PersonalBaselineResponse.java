package com.smarthealth.platform.dto;

import com.smarthealth.platform.entity.PersonalBaseline;

import java.time.LocalDateTime;

public class PersonalBaselineResponse {

    private Long id;
    private Long patientId;
    private String metricType;
    private Double baselineValue;
    private Double variability;
    private Integer observationCount;
    private LocalDateTime calculatedAt;
    private LocalDateTime updatedAt;

    public PersonalBaselineResponse() {
    }

    public static PersonalBaselineResponse fromEntity(PersonalBaseline baseline) {
        PersonalBaselineResponse response = new PersonalBaselineResponse();
        response.setId(baseline.getId());
        response.setPatientId(baseline.getPatient() != null ? baseline.getPatient().getId() : null);
        response.setMetricType(baseline.getMetricType());
        response.setBaselineValue(baseline.getBaselineValue());
        response.setVariability(baseline.getVariability());
        response.setObservationCount(baseline.getObservationCount());
        response.setCalculatedAt(baseline.getCalculatedAt());
        response.setUpdatedAt(baseline.getUpdatedAt());
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

    public Double getBaselineValue() {
        return baselineValue;
    }

    public void setBaselineValue(Double baselineValue) {
        this.baselineValue = baselineValue;
    }

    public Double getVariability() {
        return variability;
    }

    public void setVariability(Double variability) {
        this.variability = variability;
    }

    public Integer getObservationCount() {
        return observationCount;
    }

    public void setObservationCount(Integer observationCount) {
        this.observationCount = observationCount;
    }

    public LocalDateTime getCalculatedAt() {
        return calculatedAt;
    }

    public void setCalculatedAt(LocalDateTime calculatedAt) {
        this.calculatedAt = calculatedAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }
}
