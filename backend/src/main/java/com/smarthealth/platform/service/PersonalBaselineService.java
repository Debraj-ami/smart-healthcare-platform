package com.smarthealth.platform.service;

import com.smarthealth.platform.dto.PersonalBaselineResponse;
import com.smarthealth.platform.entity.HealthObservation;
import com.smarthealth.platform.entity.PersonalBaseline;
import com.smarthealth.platform.entity.User;
import com.smarthealth.platform.exception.ResourceNotFoundException;
import com.smarthealth.platform.repository.HealthObservationRepository;
import com.smarthealth.platform.repository.PersonalBaselineRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class PersonalBaselineService {

    private final PersonalBaselineRepository personalBaselineRepository;
    private final HealthObservationRepository healthObservationRepository;
    private final UserService userService;

    public PersonalBaselineService(PersonalBaselineRepository personalBaselineRepository,
                                   HealthObservationRepository healthObservationRepository,
                                   UserService userService) {
        this.personalBaselineRepository = personalBaselineRepository;
        this.healthObservationRepository = healthObservationRepository;
        this.userService = userService;
    }

    @Transactional
    public PersonalBaseline calculateOrUpdateBaseline(User patient, String metricType) {
        List<HealthObservation> observations = healthObservationRepository
                .findByPatientIdAndMetricTypeOrderByLoggedAtAsc(patient.getId(), metricType);

        if (observations.isEmpty()) {
            throw new ResourceNotFoundException("No observations found for patient " + patient.getId() + " and metric " + metricType);
        }

        int count = observations.size();
        double sum = 0.0;
        for (HealthObservation obs : observations) {
            sum += obs.getMetricValue();
        }
        double mean = sum / count;

        double variability;
        if (count <= 1) {
            variability = 0.0;
        } else {
            double sumSquaredDiffs = 0.0;
            for (HealthObservation obs : observations) {
                double diff = obs.getMetricValue() - mean;
                sumSquaredDiffs += diff * diff;
            }
            variability = Math.sqrt(sumSquaredDiffs / count);
        }

        Optional<PersonalBaseline> existing = personalBaselineRepository
                .findByPatientIdAndMetricType(patient.getId(), metricType);

        PersonalBaseline baseline;
        if (existing.isPresent()) {
            baseline = existing.get();
            baseline.setBaselineValue(mean);
            baseline.setVariability(variability);
            baseline.setObservationCount(count);
            baseline.setUpdatedAt(LocalDateTime.now());
        } else {
            baseline = new PersonalBaseline(patient, metricType, mean, variability, count);
        }

        return personalBaselineRepository.save(baseline);
    }

    @Transactional
    public PersonalBaselineResponse calculateBaseline(Long patientId, String metricType) {
        User patient = userService.getUserEntity(patientId);
        PersonalBaseline baseline = calculateOrUpdateBaseline(patient, metricType);
        return PersonalBaselineResponse.fromEntity(baseline);
    }

    @Transactional(readOnly = true)
    public List<PersonalBaselineResponse> getBaselinesByPatientId(Long patientId) {
        userService.getUserEntity(patientId);
        return personalBaselineRepository.findByPatientId(patientId)
                .stream()
                .map(PersonalBaselineResponse::fromEntity)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public PersonalBaselineResponse getBaseline(Long patientId, String metricType) {
        userService.getUserEntity(patientId);
        PersonalBaseline baseline = personalBaselineRepository
                .findByPatientIdAndMetricType(patientId, metricType)
                .orElseThrow(() -> new ResourceNotFoundException(
                        "Personal baseline not found for patient " + patientId + " and metric " + metricType));
        return PersonalBaselineResponse.fromEntity(baseline);
    }

    @Transactional(readOnly = true)
    public Optional<PersonalBaseline> getBaselineEntity(Long patientId, String metricType) {
        return personalBaselineRepository.findByPatientIdAndMetricType(patientId, metricType);
    }
}
