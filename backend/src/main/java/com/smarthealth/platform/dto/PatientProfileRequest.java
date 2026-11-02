package com.smarthealth.platform.dto;

public class PatientProfileRequest {

    private String bloodGroup;
    private Double heightCm;
    private Double weightKg;
    private String lifestyleContext;
    private String medicalHistory;

    public PatientProfileRequest() {
    }

    public PatientProfileRequest(String bloodGroup, Double heightCm, Double weightKg,
                                 String lifestyleContext, String medicalHistory) {
        this.bloodGroup = bloodGroup;
        this.heightCm = heightCm;
        this.weightKg = weightKg;
        this.lifestyleContext = lifestyleContext;
        this.medicalHistory = medicalHistory;
    }

    public String getBloodGroup() {
        return bloodGroup;
    }

    public void setBloodGroup(String bloodGroup) {
        this.bloodGroup = bloodGroup;
    }

    public Double getHeightCm() {
        return heightCm;
    }

    public void setHeightCm(Double heightCm) {
        this.heightCm = heightCm;
    }

    public Double getWeightKg() {
        return weightKg;
    }

    public void setWeightKg(Double weightKg) {
        this.weightKg = weightKg;
    }

    public String getLifestyleContext() {
        return lifestyleContext;
    }

    public void setLifestyleContext(String lifestyleContext) {
        this.lifestyleContext = lifestyleContext;
    }

    public String getMedicalHistory() {
        return medicalHistory;
    }

    public void setMedicalHistory(String medicalHistory) {
        this.medicalHistory = medicalHistory;
    }
}
