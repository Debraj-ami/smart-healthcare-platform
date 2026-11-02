package com.smarthealth.platform.service;

import com.smarthealth.platform.dto.TrendAnalysisResponse;
import com.smarthealth.platform.entity.HealthObservation;
import com.smarthealth.platform.entity.PersonalBaseline;
import com.smarthealth.platform.entity.TrendAnalysis;
import com.smarthealth.platform.entity.TrendDirection;
import com.smarthealth.platform.entity.User;
import com.smarthealth.platform.exception.ResourceNotFoundException;
import com.smarthealth.platform.repository.HealthObservationRepository;
import com.smarthealth.platform.repository.TrendAnalysisRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class TrendAnalysisService {

    private final TrendAnalysisRepository trendAnalysisRepository;
    private final HealthObservationRepository healthObservationRepository;
    private final PersonalBaselineService personalBaselineService;
    private final UserService userService;

    public TrendAnalysisService(TrendAnalysisRepository trendAnalysisRepository,
                                HealthObservationRepository healthObservationRepository,
                                PersonalBaselineService personalBaselineService,
                                UserService userService) {
        this.trendAnalysisRepository = trendAnalysisRepository;
        this.healthObservationRepository = healthObservationRepository;
        this.personalBaselineService = personalBaselineService;
        this.userService = userService;
    }

    @Transactional
    public TrendAnalysisResponse analyzeTrend(Long patientId, String metricType) {
        User patient = userService.getUserEntity(patientId);

        List<HealthObservation> observations = healthObservationRepository
                .findByPatientIdAndMetricTypeOrderByLoggedAtAsc(patientId, metricType);

        if (observations.isEmpty()) {
            throw new ResourceNotFoundException("No observations found for patient " + patientId + " and metric " + metricType);
        }

        // Retrieve existing baseline or calculate one on-the-fly
        PersonalBaseline baseline = personalBaselineService.getBaselineEntity(patientId, metricType)
                .orElseGet(() -> personalBaselineService.calculateOrUpdateBaseline(patient, metricType));

        int n = observations.size();
        HealthObservation latestObservation = observations.get(n - 1);
        double currentValue = latestObservation.getMetricValue();
        double baselineValue = baseline.getBaselineValue();
        Double variability = baseline.getVariability();

        // 1. Standardized deviation score calculation
        double deviationScore;
        if (variability != null && variability > 0.0) {
            deviationScore = (currentValue - baselineValue) / variability;
        } else {
            deviationScore = 0.0;
        }

        // 2. Trend direction & persistence calculation
        String trendDirection;
        int persistence;

        if (n == 1) {
            trendDirection = TrendDirection.STABLE.name();
            persistence = 1;
        } else {
            double last = observations.get(n - 1).getMetricValue();
            double prev = observations.get(n - 2).getMetricValue();

            if (last > prev) {
                trendDirection = TrendDirection.INCREASING.name();
                int count = 2;
                for (int i = n - 2; i >= 1; i--) {
                    if (observations.get(i).getMetricValue() > observations.get(i - 1).getMetricValue()) {
                        count++;
                    } else {
                        break;
                    }
                }
                persistence = count;
            } else if (last < prev) {
                trendDirection = TrendDirection.DECREASING.name();
                int count = 2;
                for (int i = n - 2; i >= 1; i--) {
                    if (observations.get(i).getMetricValue() < observations.get(i - 1).getMetricValue()) {
                        count++;
                    } else {
                        break;
                    }
                }
                persistence = count;
            } else {
                trendDirection = TrendDirection.STABLE.name();
                int count = 2;
                for (int i = n - 2; i >= 1; i--) {
                    if (Double.compare(observations.get(i).getMetricValue(), observations.get(i - 1).getMetricValue()) == 0) {
                        count++;
                    } else {
                        break;
                    }
                }
                persistence = count;
            }
        }

        // 3. Historical statistics snapshot
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        double sum = 0.0;
        for (HealthObservation obs : observations) {
            double v = obs.getMetricValue();
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
        double historicalMean = sum / n;

        int recentCount = Math.min(3, n);
        double recentSum = 0.0;
        for (int i = n - recentCount; i < n; i++) {
            recentSum += observations.get(i).getMetricValue();
        }
        double recentMean = recentSum / recentCount;

        // 4. Value change and percentage change snapshot
        Double previousValue = null;
        Double valueChange = null;
        Double percentageChange = null;

        if (n >= 2) {
            previousValue = observations.get(n - 2).getMetricValue();
            valueChange = currentValue - previousValue;
            if (previousValue != 0.0) {
                percentageChange = ((currentValue - previousValue) / previousValue) * 100.0;
            } else {
                percentageChange = 0.0;
            }
        } else {
            valueChange = 0.0;
            percentageChange = 0.0;
        }

        // 5. Construct and persist TrendAnalysis with immutable snapshot
        TrendAnalysis analysis = new TrendAnalysis(
                patient,
                metricType,
                baseline,
                currentValue,
                deviationScore,
                trendDirection,
                String.valueOf(persistence)
        );

        analysis.setBaselineValue(baselineValue);
        analysis.setVariability(variability);
        analysis.setDeviationFromBaseline(currentValue - baselineValue);
        analysis.setAbsoluteDeviation(Math.abs(currentValue - baselineValue));

        analysis.setPreviousValue(previousValue);
        analysis.setValueChange(valueChange);
        analysis.setPercentageChange(percentageChange);

        analysis.setObservationCount(n);
        analysis.setHistoricalMinimum(min);
        analysis.setHistoricalMaximum(max);
        analysis.setHistoricalMean(historicalMean);
        analysis.setRecentMean(recentMean);
        analysis.setDiffHistoricalMean(currentValue - historicalMean);
        analysis.setDiffRecentMean(currentValue - recentMean);

        analysis.setObservationTime(latestObservation.getLoggedAt());
        analysis.setActivityLevel(latestObservation.getActivityLevel());
        analysis.setActivityMinutes(latestObservation.getActivityMinutes());

        TrendAnalysis saved = trendAnalysisRepository.save(analysis);
        return TrendAnalysisResponse.fromEntity(saved);
    }

    @Transactional(readOnly = true)
    public List<TrendAnalysisResponse> getTrendsByPatientId(Long patientId) {
        userService.getUserEntity(patientId);
        return trendAnalysisRepository.findByPatientIdOrderByAnalyzedAtDesc(patientId)
                .stream()
                .map(TrendAnalysisResponse::fromEntity)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public List<TrendAnalysisResponse> getTrendsByPatientIdAndMetricType(Long patientId, String metricType) {
        userService.getUserEntity(patientId);
        return trendAnalysisRepository.findByPatientIdAndMetricTypeOrderByAnalyzedAtDesc(patientId, metricType)
                .stream()
                .map(TrendAnalysisResponse::fromEntity)
                .collect(Collectors.toList());
    }
}
