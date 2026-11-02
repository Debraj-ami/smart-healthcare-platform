package com.smarthealth.platform.dto;

import jakarta.validation.constraints.*;

/**
 * Request body sent by the React frontend to
 * POST /api/assessments/liver/predict
 *
 * Spring Boot validates the inputs here before forwarding to the ML service,
 * preserving the same validation constraints defined in the Python schemas.
 */
public class LiverPredictRequest {

    @NotNull(message = "patientId is required")
    private Long patientId;

    @NotNull(message = "age is required")
    @Min(value = 1, message = "age must be at least 1")
    @Max(value = 120, message = "age must be at most 120")
    private Integer age;

    @NotBlank(message = "gender is required")
    private String gender;

    @NotNull(message = "totalBilirubin is required")
    @DecimalMin(value = "0.0", inclusive = false, message = "totalBilirubin must be > 0")
    @DecimalMax(value = "100.0", message = "totalBilirubin must be ≤ 100")
    private Double totalBilirubin;

    @NotNull(message = "directBilirubin is required")
    @DecimalMin(value = "0.0", message = "directBilirubin must be ≥ 0")
    @DecimalMax(value = "50.0", message = "directBilirubin must be ≤ 50")
    private Double directBilirubin;

    @NotNull(message = "alkalinePhosphatase is required")
    @DecimalMin(value = "0.0", inclusive = false, message = "alkalinePhosphatase must be > 0")
    @DecimalMax(value = "5000.0", message = "alkalinePhosphatase must be ≤ 5000")
    private Double alkalinePhosphatase;

    @NotNull(message = "sgpt is required")
    @DecimalMin(value = "0.0", inclusive = false, message = "sgpt must be > 0")
    @DecimalMax(value = "5000.0", message = "sgpt must be ≤ 5000")
    private Double sgpt;

    @NotNull(message = "sgot is required")
    @DecimalMin(value = "0.0", inclusive = false, message = "sgot must be > 0")
    @DecimalMax(value = "5000.0", message = "sgot must be ≤ 5000")
    private Double sgot;

    @NotNull(message = "totalProteins is required")
    @DecimalMin(value = "0.0", inclusive = false, message = "totalProteins must be > 0")
    @DecimalMax(value = "20.0", message = "totalProteins must be ≤ 20")
    private Double totalProteins;

    @NotNull(message = "albumin is required")
    @DecimalMin(value = "0.0", inclusive = false, message = "albumin must be > 0")
    @DecimalMax(value = "15.0", message = "albumin must be ≤ 15")
    private Double albumin;

    @NotNull(message = "agRatio is required")
    @DecimalMin(value = "0.0", inclusive = false, message = "agRatio must be > 0")
    @DecimalMax(value = "10.0", message = "agRatio must be ≤ 10")
    private Double agRatio;

    public LiverPredictRequest() {}

    // --- Getters and Setters ---

    public Long getPatientId() { return patientId; }
    public void setPatientId(Long patientId) { this.patientId = patientId; }

    public Integer getAge() { return age; }
    public void setAge(Integer age) { this.age = age; }

    public String getGender() { return gender; }
    public void setGender(String gender) { this.gender = gender; }

    public Double getTotalBilirubin() { return totalBilirubin; }
    public void setTotalBilirubin(Double totalBilirubin) { this.totalBilirubin = totalBilirubin; }

    public Double getDirectBilirubin() { return directBilirubin; }
    public void setDirectBilirubin(Double directBilirubin) { this.directBilirubin = directBilirubin; }

    public Double getAlkalinePhosphatase() { return alkalinePhosphatase; }
    public void setAlkalinePhosphatase(Double alkalinePhosphatase) { this.alkalinePhosphatase = alkalinePhosphatase; }

    public Double getSgpt() { return sgpt; }
    public void setSgpt(Double sgpt) { this.sgpt = sgpt; }

    public Double getSgot() { return sgot; }
    public void setSgot(Double sgot) { this.sgot = sgot; }

    public Double getTotalProteins() { return totalProteins; }
    public void setTotalProteins(Double totalProteins) { this.totalProteins = totalProteins; }

    public Double getAlbumin() { return albumin; }
    public void setAlbumin(Double albumin) { this.albumin = albumin; }

    public Double getAgRatio() { return agRatio; }
    public void setAgRatio(Double agRatio) { this.agRatio = agRatio; }
}
