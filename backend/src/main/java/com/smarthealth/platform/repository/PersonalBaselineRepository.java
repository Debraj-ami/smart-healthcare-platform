package com.smarthealth.platform.repository;

import com.smarthealth.platform.entity.PersonalBaseline;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface PersonalBaselineRepository extends JpaRepository<PersonalBaseline, Long> {
    List<PersonalBaseline> findByPatientId(Long patientId);
    Optional<PersonalBaseline> findByPatientIdAndMetricType(Long patientId, String metricType);
}
