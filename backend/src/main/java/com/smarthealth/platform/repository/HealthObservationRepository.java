package com.smarthealth.platform.repository;

import com.smarthealth.platform.entity.HealthObservation;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface HealthObservationRepository extends JpaRepository<HealthObservation, Long> {
    List<HealthObservation> findByPatientIdOrderByLoggedAtDesc(Long patientId);
    List<HealthObservation> findByPatientIdAndMetricTypeOrderByLoggedAtAsc(Long patientId, String metricType);
    List<HealthObservation> findByPatientIdAndMetricTypeOrderByLoggedAtDesc(Long patientId, String metricType);
}
