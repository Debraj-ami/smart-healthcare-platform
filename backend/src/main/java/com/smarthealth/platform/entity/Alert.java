package com.smarthealth.platform.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "alerts")
public class Alert {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "patient_id", nullable = false)
    private User patient;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "assessment_id")
    private DiseaseAssessment assessment;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "trend_analysis_id")
    private TrendAnalysis trendAnalysis;

    @Column(name = "alert_type", nullable = false, length = 100)
    private String alertType;

    @Column(nullable = false, length = 50)
    private String severity;

    @Column(columnDefinition = "TEXT", nullable = false)
    private String message;

    @Column(nullable = false, length = 50)
    private String status = "ACTIVE";

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    public Alert() {
    }

    public Alert(User patient, DiseaseAssessment assessment, TrendAnalysis trendAnalysis,
                 String alertType, String severity, String message, String status) {
        this.patient = patient;
        this.assessment = assessment;
        this.trendAnalysis = trendAnalysis;
        this.alertType = alertType;
        this.severity = severity;
        this.message = message;
        this.status = status != null ? status : "ACTIVE";
    }

    @PrePersist
    protected void onCreate() {
        if (this.createdAt == null) {
            this.createdAt = LocalDateTime.now();
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

    public DiseaseAssessment getAssessment() {
        return assessment;
    }

    public void setAssessment(DiseaseAssessment assessment) {
        this.assessment = assessment;
    }

    public TrendAnalysis getTrendAnalysis() {
        return trendAnalysis;
    }

    public void setTrendAnalysis(TrendAnalysis trendAnalysis) {
        this.trendAnalysis = trendAnalysis;
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
