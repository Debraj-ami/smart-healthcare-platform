package com.smarthealth.platform.service;

import com.smarthealth.platform.dto.HealthObservationRequest;
import com.smarthealth.platform.dto.HealthObservationResponse;
import com.smarthealth.platform.entity.HealthObservation;
import com.smarthealth.platform.entity.User;
import com.smarthealth.platform.repository.HealthObservationRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class ObservationService {

    private final HealthObservationRepository observationRepository;
    private final UserService userService;
    private final PersonalBaselineService personalBaselineService;

    public ObservationService(HealthObservationRepository observationRepository,
                              UserService userService,
                              PersonalBaselineService personalBaselineService) {
        this.observationRepository = observationRepository;
        this.userService = userService;
        this.personalBaselineService = personalBaselineService;
    }

    @Transactional
    public HealthObservationResponse recordObservation(HealthObservationRequest request) {
        User patient = userService.getUserEntity(request.getPatientId());

        HealthObservation observation = new HealthObservation(
                patient,
                request.getMetricType(),
                request.getMetricValue(),
                request.getUnit(),
                request.getActivityLevel(),
                request.getActivityMinutes()
        );

        HealthObservation saved = observationRepository.save(observation);
        personalBaselineService.calculateOrUpdateBaseline(patient, request.getMetricType());
        return HealthObservationResponse.fromEntity(saved);
    }

    @Transactional(readOnly = true)
    public List<HealthObservationResponse> getObservationsByPatientId(Long patientId) {
        // Validate user existence
        userService.getUserEntity(patientId);

        return observationRepository.findByPatientIdOrderByLoggedAtDesc(patientId)
                .stream()
                .map(HealthObservationResponse::fromEntity)
                .collect(Collectors.toList());
    }
}
