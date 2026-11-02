package com.smarthealth.platform.dto;

public class HistoricalContext {

    private Integer observationCount;
    private Double historicalMinimum;
    private Double historicalMaximum;
    private Double historicalMean;
    private Double recentMean;
    private Double differenceFromHistoricalMean;
    private Double differenceFromRecentMean;

    public HistoricalContext() {
    }

    public HistoricalContext(Integer observationCount, Double historicalMinimum, Double historicalMaximum,
                             Double historicalMean, Double recentMean,
                             Double differenceFromHistoricalMean, Double differenceFromRecentMean) {
        this.observationCount = observationCount;
        this.historicalMinimum = historicalMinimum;
        this.historicalMaximum = historicalMaximum;
        this.historicalMean = historicalMean;
        this.recentMean = recentMean;
        this.differenceFromHistoricalMean = differenceFromHistoricalMean;
        this.differenceFromRecentMean = differenceFromRecentMean;
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

    public Double getDifferenceFromHistoricalMean() {
        return differenceFromHistoricalMean;
    }

    public void setDifferenceFromHistoricalMean(Double differenceFromHistoricalMean) {
        this.differenceFromHistoricalMean = differenceFromHistoricalMean;
    }

    public Double getDifferenceFromRecentMean() {
        return differenceFromRecentMean;
    }

    public void setDifferenceFromRecentMean(Double differenceFromRecentMean) {
        this.differenceFromRecentMean = differenceFromRecentMean;
    }
}
