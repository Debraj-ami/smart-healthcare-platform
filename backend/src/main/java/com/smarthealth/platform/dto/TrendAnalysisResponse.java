package com.smarthealth.platform.dto;

import com.smarthealth.platform.entity.TrendAnalysis;

import java.time.LocalDateTime;

public class TrendAnalysisResponse {

    private Long id;
    private Long patientId;
    private String metricType;
    private Long baselineId;

    private Double currentValue;
    private Double baselineValue;
    private Double variability;

    private Double deviationScore;
    private Double absoluteDeviation;
    private Double deviationFromBaseline;

    private Double previousValue;
    private Double valueChangeFromPrevious;
    private Double percentageChangeFromPrevious;

    private String trendDirection;
    private Integer persistence;

    private HistoricalContext historicalContext;
    private TemporalContext temporalContext;
    private ActivityContext activityContext;

    private LocalDateTime analyzedAt;

    public TrendAnalysisResponse() {
    }

    public static TrendAnalysisResponse fromEntity(TrendAnalysis analysis) {
        TrendAnalysisResponse response = new TrendAnalysisResponse();
        response.setId(analysis.getId());
        response.setPatientId(analysis.getPatient() != null ? analysis.getPatient().getId() : null);
        response.setMetricType(analysis.getMetricType());

        response.setBaselineId(analysis.getBaseline() != null ? analysis.getBaseline().getId() : null);
        response.setBaselineValue(analysis.getBaselineValue() != null ? analysis.getBaselineValue() :
                (analysis.getBaseline() != null ? analysis.getBaseline().getBaselineValue() : null));
        response.setVariability(analysis.getVariability() != null ? analysis.getVariability() :
                (analysis.getBaseline() != null ? analysis.getBaseline().getVariability() : null));

        Double baselineVal = response.getBaselineValue();
        Double currentVal = analysis.getCurrentValue();
        Double signedDev = analysis.getDeviationFromBaseline();
        if (signedDev == null && currentVal != null && baselineVal != null) {
            signedDev = currentVal - baselineVal;
        }
        response.setDeviationFromBaseline(signedDev);

        if (analysis.getAbsoluteDeviation() != null) {
            response.setAbsoluteDeviation(Math.abs(analysis.getAbsoluteDeviation()));
        } else if (signedDev != null) {
            response.setAbsoluteDeviation(Math.abs(signedDev));
        }

        response.setCurrentValue(analysis.getCurrentValue());
        response.setDeviationScore(analysis.getDeviationScore());
        response.setTrendDirection(analysis.getTrendDirection());

        Integer parsedPersistence = null;
        if (analysis.getPersistence() != null) {
            try {
                parsedPersistence = Integer.parseInt(analysis.getPersistence());
            } catch (NumberFormatException ignored) {
            }
        }
        response.setPersistence(parsedPersistence);

        response.setPreviousValue(analysis.getPreviousValue());
        response.setValueChangeFromPrevious(analysis.getValueChange());
        response.setPercentageChangeFromPrevious(analysis.getPercentageChange());

        // Context blocks from persisted snapshot
        if (analysis.getObservationCount() != null) {
            HistoricalContext histCtx = new HistoricalContext(
                    analysis.getObservationCount(),
                    analysis.getHistoricalMinimum(),
                    analysis.getHistoricalMaximum(),
                    analysis.getHistoricalMean(),
                    analysis.getRecentMean(),
                    analysis.getDiffHistoricalMean(),
                    analysis.getDiffRecentMean()
            );
            response.setHistoricalContext(histCtx);
        }

        if (analysis.getObservationTime() != null) {
            response.setTemporalContext(TemporalContext.fromDateTime(analysis.getObservationTime()));
        } else if (analysis.getAnalyzedAt() != null) {
            response.setTemporalContext(TemporalContext.fromDateTime(analysis.getAnalyzedAt()));
        }

        if (analysis.getActivityLevel() != null || analysis.getActivityMinutes() != null) {
            response.setActivityContext(new ActivityContext(
                    analysis.getActivityLevel() != null ? analysis.getActivityLevel() : com.smarthealth.platform.entity.ActivityLevel.UNKNOWN,
                    analysis.getActivityMinutes()
            ));
        } else if (analysis.getObservationCount() != null) {
            response.setActivityContext(new ActivityContext(com.smarthealth.platform.entity.ActivityLevel.UNKNOWN, null));
        }

        response.setAnalyzedAt(analysis.getAnalyzedAt());
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

    public Long getBaselineId() {
        return baselineId;
    }

    public void setBaselineId(Long baselineId) {
        this.baselineId = baselineId;
    }

    public Double getCurrentValue() {
        return currentValue;
    }

    public void setCurrentValue(Double currentValue) {
        this.currentValue = currentValue;
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

    public Double getDeviationScore() {
        return deviationScore;
    }

    public void setDeviationScore(Double deviationScore) {
        this.deviationScore = deviationScore;
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

    public Double getValueChangeFromPrevious() {
        return valueChangeFromPrevious;
    }

    public void setValueChangeFromPrevious(Double valueChangeFromPrevious) {
        this.valueChangeFromPrevious = valueChangeFromPrevious;
    }

    public Double getPercentageChangeFromPrevious() {
        return percentageChangeFromPrevious;
    }

    public void setPercentageChangeFromPrevious(Double percentageChangeFromPrevious) {
        this.percentageChangeFromPrevious = percentageChangeFromPrevious;
    }

    public String getTrendDirection() {
        return trendDirection;
    }

    public void setTrendDirection(String trendDirection) {
        this.trendDirection = trendDirection;
    }

    public Integer getPersistence() {
        return persistence;
    }

    public void setPersistence(Integer persistence) {
        this.persistence = persistence;
    }

    public HistoricalContext getHistoricalContext() {
        return historicalContext;
    }

    public void setHistoricalContext(HistoricalContext historicalContext) {
        this.historicalContext = historicalContext;
    }

    public TemporalContext getTemporalContext() {
        return temporalContext;
    }

    public void setTemporalContext(TemporalContext temporalContext) {
        this.temporalContext = temporalContext;
    }

    public ActivityContext getActivityContext() {
        return activityContext;
    }

    public void setActivityContext(ActivityContext activityContext) {
        this.activityContext = activityContext;
    }

    public LocalDateTime getAnalyzedAt() {
        return analyzedAt;
    }

    public void setAnalyzedAt(LocalDateTime analyzedAt) {
        this.analyzedAt = analyzedAt;
    }
}
