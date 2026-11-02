package com.smarthealth.platform.repository;

import com.smarthealth.platform.entity.DiseaseAssessment;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface DiseaseAssessmentRepository extends JpaRepository<DiseaseAssessment, Long> {
    List<DiseaseAssessment> findByPatientIdOrderByAssessedAtDesc(Long patientId);
    List<DiseaseAssessment> findByPatientIdAndDiseaseTypeOrderByAssessedAtDesc(Long patientId, com.smarthealth.platform.entity.DiseaseType diseaseType);
}

