package com.smarthealth.platform.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "trend_analyses")
public class TrendAnalysis {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "patient_id", nullable = false)
    private User patient;

    @Column(name = "metric_type", nullable = false, length = 100)
    private String metricType;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "baseline_id")
    private PersonalBaseline baseline;

    @Column(name = "current_value", nullable = false)
    private Double currentValue;

    @Column(name = "deviation_score")
    private Double deviationScore;

    @Column(name = "trend_direction", length = 50)
    private String trendDirection;

    @Column(length = 50)
    private String persistence;

    // --- Snapshot fields preserved at analysis time ---

    @Column(name = "baseline_value")
    private Double baselineValue;

    private Double variability;

    @Column(name = "absolute_deviation")
    private Double absoluteDeviation;

    @Column(name = "deviation_from_baseline")
    private Double deviationFromBaseline;

    @Column(name = "previous_value")
    private Double previousValue;

    @Column(name = "value_change")
    private Double valueChange;

    @Column(name = "percentage_change")
    private Double percentageChange;

    @Column(name = "observation_count")
    private Integer observationCount;

    @Column(name = "historical_minimum")
    private Double historicalMinimum;

    @Column(name = "historical_maximum")
    private Double historicalMaximum;

    @Column(name = "historical_mean")
    private Double historicalMean;

    @Column(name = "recent_mean")
    private Double recentMean;

    @Column(name = "diff_historical_mean")
    private Double diffHistoricalMean;

    @Column(name = "diff_recent_mean")
    private Double diffRecentMean;

    @Column(name = "observation_time")
    private LocalDateTime observationTime;

    @Enumerated(EnumType.STRING)
    @Column(name = "activity_level", length = 50)
    private ActivityLevel activityLevel;

    @Column(name = "activity_minutes")
    private Integer activityMinutes;

    @Column(name = "analyzed_at", nullable = false)
    private LocalDateTime analyzedAt;

    public TrendAnalysis() {
    }

    public TrendAnalysis(User patient, String metricType, PersonalBaseline baseline,
                         Double currentValue, Double deviationScore, String trendDirection,
                         String persistence) {
        this.patient = patient;
        this.metricType = metricType;
        this.baseline = baseline;
        this.currentValue = currentValue;
        this.deviationScore = deviationScore;
        this.trendDirection = trendDirection;
        this.persistence = persistence;
    }

    @PrePersist
    protected void onCreate() {
        if (this.analyzedAt == null) {
            this.analyzedAt = LocalDateTime.now();
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

    public PersonalBaseline getBaseline() {
        return baseline;
    }

    public void setBaseline(PersonalBaseline baseline) {
        this.baseline = baseline;
    }

    public Double getCurrentValue() {
        return currentValue;
    }

    public void setCurrentValue(Double currentValue) {
        this.currentValue = currentValue;
    }

    public Double getDeviationScore() {
        return deviationScore;
    }

    public void setDeviationScore(Double deviationScore) {
        this.deviationScore = deviationScore;
    }

    public String getTrendDirection() {
        return trendDirection;
    }

    public void setTrendDirection(String trendDirection) {
        this.trendDirection = trendDirection;
    }

    public String getPersistence() {
        return persistence;
    }

    public void setPersistence(String persistence) {
        this.persistence = persistence;
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

    public Double getAbsoluteDeviation() {
        return absoluteDeviation;
    }

    public void setAbsoluteDeviation(Double absoluteDeviation) {
        this.absoluteDeviation = absoluteDeviation != null ? Math.abs(absoluteDeviation) : null;
    }

    public Double getDeviationFromBaseline() {
        return deviationFromBaseline;
    }

    public void setDeviationFromBaseline(Double deviationFromBaseline) {
        this.deviationFromBaseline = deviationFromBaseline;
    }

    public Double getPreviousValue() {
        return previousValue;
    }

    public void setPreviousValue(Double previousValue) {
        this.previousValue = previousValue;
    }

    public Double getValueChange() {
        return valueChange;
    }

    public void setValueChange(Double valueChange) {
        this.valueChange = valueChange;
    }

    public Double getPercentageChange() {
        return percentageChange;
    }

    public void setPercentageChange(Double percentageChange) {
        this.percentageChange = percentageChange;
    }

    public Integer getObservationCount() {
        return observationCount;
    }

    public void setObservationCount(Integer observationCount) {
        this.observationCount = observationCount;
    }

    public Double getHistoricalMinimum() {
        return historicalMinimum;
    }

    public void setHistoricalMinimum(Double historicalMinimum) {
        this.historicalMinimum = historicalMinimum;
    }

    public Double getHistoricalMaximum() {
        return historicalMaximum;
    }

    public void setHistoricalMaximum(Double historicalMaximum) {
        this.historicalMaximum = historicalMaximum;
    }

    public Double getHistoricalMean() {
        return historicalMean;
    }

    public void setHistoricalMean(Double historicalMean) {
        this.historicalMean = historicalMean;
    }

    public Double getRecentMean() {
        return recentMean;
    }

    public void setRecentMean(Double recentMean) {
        this.recentMean = recentMean;
    }

    public Double getDiffHistoricalMean() {
        return diffHistoricalMean;
    }

    public void setDiffHistoricalMean(Double diffHistoricalMean) {
        this.diffHistoricalMean = diffHistoricalMean;
    }

    public Double getDiffRecentMean() {
        return diffRecentMean;
    }

    public void setDiffRecentMean(Double diffRecentMean) {
        this.diffRecentMean = diffRecentMean;
    }

    public LocalDateTime getObservationTime() {
        return observationTime;
    }

    public void setObservationTime(LocalDateTime observationTime) {
        this.observationTime = observationTime;
    }

    public ActivityLevel getActivityLevel() {
        return activityLevel;
    }

    public void setActivityLevel(ActivityLevel activityLevel) {
        this.activityLevel = activityLevel;
    }

    public Integer getActivityMinutes() {
        return activityMinutes;
    }

    public void setActivityMinutes(Integer activityMinutes) {
        this.activityMinutes = activityMinutes;
    }

    public LocalDateTime getAnalyzedAt() {
        return analyzedAt;
    }

    public void setAnalyzedAt(LocalDateTime analyzedAt) {
        this.analyzedAt = analyzedAt;
    }
}
