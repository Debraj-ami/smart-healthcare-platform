package com.smarthealth.platform.controller;

import com.smarthealth.platform.dto.PatientProfileRequest;
import com.smarthealth.platform.dto.PatientProfileResponse;
import com.smarthealth.platform.service.PatientProfileService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/patients")
@Tag(name = "Patients", description = "Patient profiles and clinical background")
public class PatientController {

    private final PatientProfileService profileService;

    public PatientController(PatientProfileService profileService) {
        this.profileService = profileService;
    }

    @GetMapping("/{id}")
    @Operation(summary = "Retrieve a patient profile by user/patient ID")
    public ResponseEntity<PatientProfileResponse> getPatientProfile(@PathVariable Long id) {
        PatientProfileResponse response = profileService.getProfileByUserId(id);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/{id}/profile")
    @Operation(summary = "Create or update patient profile for a user ID")
    public ResponseEntity<PatientProfileResponse> createOrUpdateProfile(
            @PathVariable Long id,
            @RequestBody PatientProfileRequest request) {
        PatientProfileResponse response = profileService.createOrUpdateProfile(id, request);
        return ResponseEntity.ok(response);
    }
}
