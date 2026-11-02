package com.smarthealth.platform.service;

import com.smarthealth.platform.dto.DiseaseAssessmentRequest;
import com.smarthealth.platform.dto.DiseaseAssessmentResponse;
import com.smarthealth.platform.entity.DiseaseAssessment;
import com.smarthealth.platform.entity.User;
import com.smarthealth.platform.repository.DiseaseAssessmentRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class AssessmentService {

    private final DiseaseAssessmentRepository assessmentRepository;
    private final UserService userService;

    public AssessmentService(DiseaseAssessmentRepository assessmentRepository, UserService userService) {
        this.assessmentRepository = assessmentRepository;
        this.userService = userService;
    }

    @Transactional
    public DiseaseAssessmentResponse recordAssessment(DiseaseAssessmentRequest request) {
        User patient = userService.getUserEntity(request.getPatientId());

        DiseaseAssessment assessment = new DiseaseAssessment(
                patient,
                request.getDiseaseType(),
                request.getRiskScore(),
                request.getRiskLevel(),
                request.getInputFeatures()
        );

        DiseaseAssessment saved = assessmentRepository.save(assessment);
        return DiseaseAssessmentResponse.fromEntity(saved);
    }

    @Transactional(readOnly = true)
    public List<DiseaseAssessmentResponse> getAssessmentsByPatientId(Long patientId) {
        // Validate user existence
        userService.getUserEntity(patientId);

        return assessmentRepository.findByPatientIdOrderByAssessedAtDesc(patientId)
                .stream()
                .map(DiseaseAssessmentResponse::fromEntity)
                .collect(Collectors.toList());
    }
}
