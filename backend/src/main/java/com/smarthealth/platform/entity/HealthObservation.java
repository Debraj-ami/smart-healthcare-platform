package com.smarthealth.platform.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "health_observations")
public class HealthObservation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "patient_id", nullable = false)
    private User patient;

    @Column(name = "metric_type", nullable = false, length = 100)
    private String metricType;

    @Column(name = "metric_value", nullable = false)
    private Double metricValue;

    @Column(length = 50)
    private String unit;

    @Column(name = "logged_at", nullable = false)
    private LocalDateTime loggedAt;

    @Enumerated(EnumType.STRING)
    @Column(name = "activity_level", length = 50)
    private ActivityLevel activityLevel = ActivityLevel.UNKNOWN;

    @Column(name = "activity_minutes")
    private Integer activityMinutes;

    public HealthObservation() {
    }

    public HealthObservation(User patient, String metricType, Double metricValue, String unit) {
        this.patient = patient;
        this.metricType = metricType;
        this.metricValue = metricValue;
        this.unit = unit;
        this.activityLevel = ActivityLevel.UNKNOWN;
    }

    public HealthObservation(User patient, String metricType, Double metricValue, String unit,
                             ActivityLevel activityLevel, Integer activityMinutes) {
        this.patient = patient;
        this.metricType = metricType;
        this.metricValue = metricValue;
        this.unit = unit;
        this.activityLevel = activityLevel != null ? activityLevel : ActivityLevel.UNKNOWN;
        this.activityMinutes = activityMinutes;
    }

    @PrePersist
    protected void onCreate() {
        if (this.loggedAt == null) {
            this.loggedAt = LocalDateTime.now();
        }
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

    public ActivityLevel getActivityLevel() {
        return activityLevel;
    }

    public void setActivityLevel(ActivityLevel activityLevel) {
        this.activityLevel = activityLevel != null ? activityLevel : ActivityLevel.UNKNOWN;
    }

    public Integer getActivityMinutes() {
        return activityMinutes;
    }

    public void setActivityMinutes(Integer activityMinutes) {
        this.activityMinutes = activityMinutes;
    }
}
