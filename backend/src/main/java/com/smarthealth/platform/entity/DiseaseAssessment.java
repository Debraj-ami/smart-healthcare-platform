package com.smarthealth.platform.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "disease_assessments")
public class DiseaseAssessment {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "patient_id", nullable = false)
    private User patient;

    @Enumerated(EnumType.STRING)
    @Column(name = "disease_type", nullable = false, length = 50)
    private DiseaseType diseaseType;

    @Column(name = "risk_score", nullable = false)
    private Double riskScore;

    @Enumerated(EnumType.STRING)
    @Column(name = "risk_level", length = 50)
    private RiskLevel riskLevel;

    @Column(name = "input_features", columnDefinition = "TEXT")
    private String inputFeatures;

    @Column(name = "assessed_at", nullable = false)
    private LocalDateTime assessedAt;

    public DiseaseAssessment() {
    }

    public DiseaseAssessment(User patient, DiseaseType diseaseType, Double riskScore,
                             RiskLevel riskLevel, String inputFeatures) {
        this.patient = patient;
        this.diseaseType = diseaseType;
        this.riskScore = riskScore;
        this.riskLevel = riskLevel;
        this.inputFeatures = inputFeatures;
    }

    @PrePersist
    protected void onCreate() {
        if (this.assessedAt == null) {
            this.assessedAt = LocalDateTime.now();
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
