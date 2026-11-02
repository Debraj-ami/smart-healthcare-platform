package com.smarthealth.platform.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "personal_baselines", uniqueConstraints = {
        @UniqueConstraint(name = "uk_baseline_patient_metric", columnNames = {"patient_id", "metric_type"})
})
public class PersonalBaseline {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "patient_id", nullable = false)
    private User patient;

    @Column(name = "metric_type", nullable = false, length = 100)
    private String metricType;

    @Column(name = "baseline_value", nullable = false)
    private Double baselineValue;

    private Double variability;

    @Column(name = "observation_count")
    private Integer observationCount = 0;

    @Column(name = "calculated_at", nullable = false)
    private LocalDateTime calculatedAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    public PersonalBaseline() {
    }

    public PersonalBaseline(User patient, String metricType, Double baselineValue,
                            Double variability, Integer observationCount) {
        this.patient = patient;
        this.metricType = metricType;
        this.baselineValue = baselineValue;
        this.variability = variability;
        this.observationCount = observationCount != null ? observationCount : 0;
    }

    @PrePersist
    protected void onCreate() {
        this.calculatedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public User getPatient() {
        return patient;
    }

    public void setPatient(User patient) {
        this.patient = patient;
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
