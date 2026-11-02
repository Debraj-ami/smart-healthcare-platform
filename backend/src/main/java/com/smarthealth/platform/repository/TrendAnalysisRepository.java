package com.smarthealth.platform.repository;

import com.smarthealth.platform.entity.TrendAnalysis;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface TrendAnalysisRepository extends JpaRepository<TrendAnalysis, Long> {
    List<TrendAnalysis> findByPatientIdOrderByAnalyzedAtDesc(Long patientId);
    List<TrendAnalysis> findByPatientIdAndMetricTypeOrderByAnalyzedAtDesc(Long patientId, String metricType);
}
