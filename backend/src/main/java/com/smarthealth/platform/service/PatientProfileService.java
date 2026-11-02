package com.smarthealth.platform.service;

import com.smarthealth.platform.dto.PatientProfileRequest;
import com.smarthealth.platform.dto.PatientProfileResponse;
import com.smarthealth.platform.entity.PatientProfile;
import com.smarthealth.platform.entity.User;
import com.smarthealth.platform.exception.ResourceNotFoundException;
import com.smarthealth.platform.repository.PatientProfileRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class PatientProfileService {

    private final PatientProfileRepository profileRepository;
    private final UserService userService;

    public PatientProfileService(PatientProfileRepository profileRepository, UserService userService) {
        this.profileRepository = profileRepository;
        this.userService = userService;
    }

    @Transactional(readOnly = true)
    public PatientProfileResponse getProfileByUserId(Long userId) {
        User user = userService.getUserEntity(userId);
        return profileRepository.findByUserId(userId)
                .map(PatientProfileResponse::fromEntity)
                .orElseGet(() -> {
                    PatientProfileResponse response = new PatientProfileResponse();
                    response.setUserId(user.getId());
                    return response;
                });
    }

    @Transactional
    public PatientProfileResponse createOrUpdateProfile(Long userId, PatientProfileRequest request) {
        User user = userService.getUserEntity(userId);

        PatientProfile profile = profileRepository.findByUserId(userId)
                .orElse(new PatientProfile());

        profile.setUser(user);
        profile.setBloodGroup(request.getBloodGroup());
        profile.setHeightCm(request.getHeightCm());
        profile.setWeightKg(request.getWeightKg());
        profile.setLifestyleContext(request.getLifestyleContext());
        profile.setMedicalHistory(request.getMedicalHistory());

        PatientProfile saved = profileRepository.save(profile);
        return PatientProfileResponse.fromEntity(saved);
    }
}
