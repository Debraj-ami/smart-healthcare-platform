package com.smarthealth.platform.dto;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.*;

/**
 * Request body for Heart Disease prediction sent to
 * POST /api/assessments/heart/predict
 *
 * Contains the exact 13 clinical predictor features established by the
 * Cleveland Heart Disease ML lifecycle:
 * age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
 *
 * The target variable 'num' is strictly forbidden and rejected.
 * Nullable features 'ca' and 'thal' accept null values, which are imputed by
 * the production preprocessing pipeline in the ML service.
 */
public class HeartPredictRequest {

    /**
     * Optional patient ID. If provided, the assessment record is persisted
     * to the disease_assessments table. If omitted, pure stateless inference is performed.
     */
    private Long patientId;

    @NotNull(message = "age is required")
    @DecimalMin(value = "1.0", message = "age must be at least 1")
    @DecimalMax(value = "120.0", message = "age must be at most 120")
    private Double age;

    @NotNull(message = "sex is required")
    private Double sex;

    @NotNull(message = "cp is required")
    private Double cp;

    @NotNull(message = "trestbps is required")
    @DecimalMin(value = "50.0", message = "trestbps must be at least 50")
    @DecimalMax(value = "300.0", message = "trestbps must be at most 300")
    private Double trestbps;

    @NotNull(message = "chol is required")
    @DecimalMin(value = "50.0", message = "chol must be at least 50")
    @DecimalMax(value = "800.0", message = "chol must be at most 800")
    private Double chol;

    @NotNull(message = "fbs is required")
    private Double fbs;

    @NotNull(message = "restecg is required")
    private Double restecg;

    @NotNull(message = "thalach is required")
    @DecimalMin(value = "50.0", message = "thalach must be at least 50")
    @DecimalMax(value = "250.0", message = "thalach must be at most 250")
    private Double thalach;

    @NotNull(message = "exang is required")
    private Double exang;

    @NotNull(message = "oldpeak is required")
    @DecimalMin(value = "0.0", message = "oldpeak must be at least 0.0")
    @DecimalMax(value = "15.0", message = "oldpeak must be at most 15.0")
    private Double oldpeak;

    @NotNull(message = "slope is required")
    private Double slope;

    /**
     * Number of major vessels (0.0-3.0) colored by fluoroscopy.
     * Nullable per Phase 4 contract (imputed with median).
     */
    private Double ca;

    /**
     * Thallium scintigraphy defect (3.0: normal, 6.0: fixed, 7.0: reversible).
     * Nullable per Phase 4 contract (imputed with mode).
     */
    private Double thal;

    public HeartPredictRequest() {}

    public HeartPredictRequest(Double age, Double sex, Double cp, Double trestbps, Double chol,
                               Double fbs, Double restecg, Double thalach, Double exang,
                               Double oldpeak, Double slope, Double ca, Double thal) {
        this.age = age;
        this.sex = sex;
        this.cp = cp;
        this.trestbps = trestbps;
        this.chol = chol;
        this.fbs = fbs;
        this.restecg = restecg;
        this.thalach = thalach;
        this.exang = exang;
        this.oldpeak = oldpeak;
        this.slope = slope;
        this.ca = ca;
        this.thal = thal;
    }

    // --- Validation Guards for Categorical Discrete Sets ---

    @AssertTrue(message = "sex must be 0 (Female) or 1 (Male)")
    @JsonIgnore
    public boolean isSexValid() {
        return sex == null || sex == 0.0 || sex == 1.0;
    }

    @AssertTrue(message = "cp must be 1.0, 2.0, 3.0, or 4.0")
    @JsonIgnore
    public boolean isCpValid() {
        return cp == null || cp == 1.0 || cp == 2.0 || cp == 3.0 || cp == 4.0;
    }

    @AssertTrue(message = "fbs must be 0 (<= 120 mg/dl) or 1 (> 120 mg/dl)")
    @JsonIgnore
    public boolean isFbsValid() {
        return fbs == null || fbs == 0.0 || fbs == 1.0;
    }

    @AssertTrue(message = "restecg must be 0.0, 1.0, or 2.0")
    @JsonIgnore
    public boolean isRestecgValid() {
        return restecg == null || restecg == 0.0 || restecg == 1.0 || restecg == 2.0;
    }

    @AssertTrue(message = "exang must be 0 (No) or 1 (Yes)")
    @JsonIgnore
    public boolean isExangValid() {
        return exang == null || exang == 0.0 || exang == 1.0;
    }

    @AssertTrue(message = "slope must be 1.0, 2.0, or 3.0")
    @JsonIgnore
    public boolean isSlopeValid() {
        return slope == null || slope == 1.0 || slope == 2.0 || slope == 3.0;
    }

    @AssertTrue(message = "ca must be null or 0.0, 1.0, 2.0, 3.0")
    @JsonIgnore
    public boolean isCaValid() {
        return ca == null || ca == 0.0 || ca == 1.0 || ca == 2.0 || ca == 3.0;
    }

    @AssertTrue(message = "thal must be null or 3.0 (normal), 6.0 (fixed), 7.0 (reversible)")
    @JsonIgnore
    public boolean isThalValid() {
        return thal == null || thal == 3.0 || thal == 6.0 || thal == 7.0;
    }

    // --- Strict Rejection of Target Field 'num' ---

    @JsonProperty("num")
    public void setNum(Object num) {
        throw new IllegalArgumentException("Target variable 'num' is not accepted in prediction requests.");
    }

    // --- Getters and Setters ---

    public Long getPatientId() { return patientId; }
    public void setPatientId(Long patientId) { this.patientId = patientId; }

    public Double getAge() { return age; }
    public void setAge(Double age) { this.age = age; }

    public Double getSex() { return sex; }
    public void setSex(Double sex) { this.sex = sex; }

    public Double getCp() { return cp; }
    public void setCp(Double cp) { this.cp = cp; }

    public Double getTrestbps() { return trestbps; }
    public void setTrestbps(Double trestbps) { this.trestbps = trestbps; }

    public Double getChol() { return chol; }
    public void setChol(Double chol) { this.chol = chol; }

    public Double getFbs() { return fbs; }
    public void setFbs(Double fbs) { this.fbs = fbs; }

    public Double getRestecg() { return restecg; }
    public void setRestecg(Double restecg) { this.restecg = restecg; }

    public Double getThalach() { return thalach; }
    public void setThalach(Double thalach) { this.thalach = thalach; }

    public Double getExang() { return exang; }
    public void setExang(Double exang) { this.exang = exang; }

    public Double getOldpeak() { return oldpeak; }
    public void setOldpeak(Double oldpeak) { this.oldpeak = oldpeak; }

    public Double getSlope() { return slope; }
    public void setSlope(Double slope) { this.slope = slope; }

    public Double getCa() { return ca; }
    public void setCa(Double ca) { this.ca = ca; }

    public Double getThal() { return thal; }
    public void setThal(Double thal) { this.thal = thal; }
}
