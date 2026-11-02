package com.smarthealth.platform.service;

import com.smarthealth.platform.dto.AlertResponse;
import com.smarthealth.platform.repository.AlertRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class AlertService {

    private final AlertRepository alertRepository;
    private final UserService userService;

    public AlertService(AlertRepository alertRepository, UserService userService) {
        this.alertRepository = alertRepository;
        this.userService = userService;
    }

    @Transactional(readOnly = true)
    public List<AlertResponse> getAlertsByPatientId(Long patientId) {
        // Validate user existence
        userService.getUserEntity(patientId);

        return alertRepository.findByPatientIdOrderByCreatedAtDesc(patientId)
                .stream()
                .map(AlertResponse::fromEntity)
                .collect(Collectors.toList());
    }
}
