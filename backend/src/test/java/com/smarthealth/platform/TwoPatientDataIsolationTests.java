package com.smarthealth.platform;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.smarthealth.platform.dto.HealthObservationRequest;
import com.smarthealth.platform.dto.UserCreateRequest;
import com.smarthealth.platform.dto.UserResponse;
import com.smarthealth.platform.entity.*;
import com.smarthealth.platform.repository.DiseaseAssessmentRepository;
import com.smarthealth.platform.repository.PersonalBaselineRepository;
import com.smarthealth.platform.repository.UserRepository;
import com.smarthealth.platform.service.UserService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

import static org.hamcrest.Matchers.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Stage 8 Final Closure — GAP 2: Real Two-Patient Data Isolation Integration Test.
 *
 * Verifies strict data isolation across real service boundaries for two distinct test patients:
 * - Patient A and Patient B have distinct accounts, observations, baselines, and Heart assessments.
 * - Patient A endpoints MUST NOT leak any Patient B data (IDs, metrics, probabilities, SHAP data).
 * - Patient B endpoints MUST NOT leak any Patient A data.
 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class TwoPatientDataIsolationTests {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private UserService userService;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private DiseaseAssessmentRepository assessmentRepository;

    @Autowired
    private PersonalBaselineRepository baselineRepository;

    private Long patientAId;
    private Long patientBId;
    private Long assessmentAId;
    private Long assessmentBId;

    @BeforeEach
    void setUp() throws Exception {
        long timestamp = System.currentTimeMillis();

        // 1. Create Patient A
        UserResponse userA = userService.createUser(new UserCreateRequest(
                "patientA." + timestamp + "@validation.test",
                "PasswordA123!",
                "Alice",
                "PatientA",
                LocalDate.of(1980, 5, 20),
                "Female",
                UserRole.PATIENT
        ));
        patientAId = userA.getId();
        User entityA = userRepository.findById(patientAId).orElseThrow();

        // 2. Create Patient B
        UserResponse userB = userService.createUser(new UserCreateRequest(
                "patientB." + timestamp + "@validation.test",
                "PasswordB123!",
                "Bob",
                "PatientB",
                LocalDate.of(1965, 11, 14),
                "Male",
                UserRole.PATIENT
        ));
        patientBId = userB.getId();
        User entityB = userRepository.findById(patientBId).orElseThrow();

        // 3. Insert unique, distinguishable observations for Patient A
        HealthObservationRequest obsA1 = new HealthObservationRequest(patientAId, "blood_glucose", 112.5, "mg/dL");
        HealthObservationRequest obsA2 = new HealthObservationRequest(patientAId, "systolic_blood_pressure", 118.0, "mm Hg");
        mockMvc.perform(post("/api/observations").contentType(MediaType.APPLICATION_JSON).content(objectMapper.writeValueAsString(obsA1)))
                .andExpect(status().isCreated());
        mockMvc.perform(post("/api/observations").contentType(MediaType.APPLICATION_JSON).content(objectMapper.writeValueAsString(obsA2)))
                .andExpect(status().isCreated());

        // 4. Insert unique, distinguishable observations for Patient B
        HealthObservationRequest obsB1 = new HealthObservationRequest(patientBId, "blood_glucose", 185.0, "mg/dL");
        HealthObservationRequest obsB2 = new HealthObservationRequest(patientBId, "total_cholesterol", 265.0, "mg/dL");
        mockMvc.perform(post("/api/observations").contentType(MediaType.APPLICATION_JSON).content(objectMapper.writeValueAsString(obsB1)))
                .andExpect(status().isCreated());
        mockMvc.perform(post("/api/observations").contentType(MediaType.APPLICATION_JSON).content(objectMapper.writeValueAsString(obsB2)))
                .andExpect(status().isCreated());

        // 5. Insert unique, distinguishable Heart assessment for Patient A
        String featuresA = objectMapper.writeValueAsString(Map.of(
                "probability", 0.1234,
                "prediction", 0,
                "threshold", 0.31,
                "modelName", "Logistic Regression (Robust L1)",
                "explanationAvailable", true,
                "inputSummary", Map.of("age", 45, "chol", 195, "isolationMarker", "PATIENT_A_EXCLUSIVE_MARKER"),
                "positiveContributors", List.of(Map.of("feature", "age", "contribution", 0.1111, "transformedFeature", "num__age")),
                "negativeContributors", List.of()
        ));
        DiseaseAssessment daA = new DiseaseAssessment(entityA, DiseaseType.HEART_DISEASE, 0.1234, RiskLevel.LOW, featuresA);
        DiseaseAssessment savedA = assessmentRepository.save(daA);
        assessmentAId = savedA.getId();

        // 6. Insert unique, distinguishable Heart assessment for Patient B
        String featuresB = objectMapper.writeValueAsString(Map.of(
                "probability", 0.8765,
                "prediction", 1,
                "threshold", 0.31,
                "modelName", "Logistic Regression (Robust L1)",
                "explanationAvailable", true,
                "inputSummary", Map.of("age", 68, "chol", 290, "isolationMarker", "PATIENT_B_EXCLUSIVE_MARKER"),
                "positiveContributors", List.of(Map.of("feature", "chol", "contribution", 0.7777, "transformedFeature", "num__chol")),
                "negativeContributors", List.of()
        ));
        DiseaseAssessment daB = new DiseaseAssessment(entityB, DiseaseType.HEART_DISEASE, 0.8765, RiskLevel.HIGH, featuresB);
        DiseaseAssessment savedB = assessmentRepository.save(daB);
        assessmentBId = savedB.getId();

        // 7. Personal baselines are automatically established by the platform when observations are logged.
    }

    @Test
    @DisplayName("1. Observations endpoint strictly isolates Patient A and Patient B data")
    void testObservationIsolation() throws Exception {
        // Query Patient A observations
        mockMvc.perform(get("/api/observations/patient/" + patientAId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(2)))
                .andExpect(jsonPath("$[*].patientId", everyItem(is(patientAId.intValue()))))
                .andExpect(jsonPath("$[*].metricValue", hasItems(112.5, 118.0)))
                // Explicit assertion: Patient A response MUST NOT contain Patient B observations
                .andExpect(jsonPath("$[*].metricValue", not(hasItem(185.0))))
                .andExpect(jsonPath("$[*].metricValue", not(hasItem(265.0))))
                .andExpect(jsonPath("$[*].metricType", not(hasItem("total_cholesterol"))));

        // Query Patient B observations
        mockMvc.perform(get("/api/observations/patient/" + patientBId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(2)))
                .andExpect(jsonPath("$[*].patientId", everyItem(is(patientBId.intValue()))))
                .andExpect(jsonPath("$[*].metricValue", hasItems(185.0, 265.0)))
                // Explicit assertion: Patient B response MUST NOT contain Patient A observations
                .andExpect(jsonPath("$[*].metricValue", not(hasItem(112.5))))
                .andExpect(jsonPath("$[*].metricValue", not(hasItem(118.0))))
                .andExpect(jsonPath("$[*].metricType", not(hasItem("systolic_blood_pressure"))));
    }

    @Test
    @DisplayName("2. Heart Assessment History endpoint strictly isolates Patient A and Patient B assessments")
    void testHeartAssessmentHistoryIsolation() throws Exception {
        // Query Patient A Heart assessment history
        mockMvc.perform(get("/api/assessments/heart/patient/" + patientAId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(1)))
                .andExpect(jsonPath("$[0].id").value(assessmentAId))
                .andExpect(jsonPath("$[0].patientId").value(patientAId))
                .andExpect(jsonPath("$[0].riskScore").value(0.1234))
                .andExpect(jsonPath("$[0].riskLevel").value("LOW"))
                .andExpect(jsonPath("$[0].inputFeatures", containsString("PATIENT_A_EXCLUSIVE_MARKER")))
                .andExpect(jsonPath("$[0].inputFeatures", containsString("0.1111")))
                // Explicit assertion: Patient A MUST NOT receive Patient B assessment data
                .andExpect(jsonPath("$[0].id", not(is(assessmentBId.intValue()))))
                .andExpect(jsonPath("$[0].riskScore", not(is(0.8765))))
                .andExpect(jsonPath("$[0].inputFeatures", not(containsString("PATIENT_B_EXCLUSIVE_MARKER"))))
                .andExpect(jsonPath("$[0].inputFeatures", not(containsString("0.7777"))));

        // Query Patient B Heart assessment history
        mockMvc.perform(get("/api/assessments/heart/patient/" + patientBId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(1)))
                .andExpect(jsonPath("$[0].id").value(assessmentBId))
                .andExpect(jsonPath("$[0].patientId").value(patientBId))
                .andExpect(jsonPath("$[0].riskScore").value(0.8765))
                .andExpect(jsonPath("$[0].riskLevel").value("HIGH"))
                .andExpect(jsonPath("$[0].inputFeatures", containsString("PATIENT_B_EXCLUSIVE_MARKER")))
                .andExpect(jsonPath("$[0].inputFeatures", containsString("0.7777")))
                // Explicit assertion: Patient B MUST NOT receive Patient A assessment data
                .andExpect(jsonPath("$[0].id", not(is(assessmentAId.intValue()))))
                .andExpect(jsonPath("$[0].riskScore", not(is(0.1234))))
                .andExpect(jsonPath("$[0].inputFeatures", not(containsString("PATIENT_A_EXCLUSIVE_MARKER"))))
                .andExpect(jsonPath("$[0].inputFeatures", not(containsString("0.1111"))));
    }

    @Test
    @DisplayName("3. Personal Baselines endpoint strictly isolates Patient A and Patient B baselines")
    void testBaselineIsolation() throws Exception {
        // Query Patient A baselines (has blood_glucose and systolic_blood_pressure)
        mockMvc.perform(get("/api/baselines/patient/" + patientAId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(2)))
                .andExpect(jsonPath("$[*].patientId", everyItem(is(patientAId.intValue()))))
                .andExpect(jsonPath("$[*].metricType", hasItems("blood_glucose", "systolic_blood_pressure")))
                .andExpect(jsonPath("$[*].baselineValue", hasItems(112.5, 118.0)))
                // Explicit assertion: Patient A MUST NOT receive Patient B baseline
                .andExpect(jsonPath("$[*].metricType", not(hasItem("total_cholesterol"))))
                .andExpect(jsonPath("$[*].baselineValue", not(hasItem(265.0))))
                .andExpect(jsonPath("$[*].baselineValue", not(hasItem(185.0))));

        // Query Patient B baselines (has blood_glucose and total_cholesterol)
        mockMvc.perform(get("/api/baselines/patient/" + patientBId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(2)))
                .andExpect(jsonPath("$[*].patientId", everyItem(is(patientBId.intValue()))))
                .andExpect(jsonPath("$[*].metricType", hasItems("blood_glucose", "total_cholesterol")))
                .andExpect(jsonPath("$[*].baselineValue", hasItems(185.0, 265.0)))
                // Explicit assertion: Patient B MUST NOT receive Patient A baseline
                .andExpect(jsonPath("$[*].metricType", not(hasItem("systolic_blood_pressure"))))
                .andExpect(jsonPath("$[*].baselineValue", not(hasItem(112.5))))
                .andExpect(jsonPath("$[*].baselineValue", not(hasItem(118.0))));
    }

    @Test
    @DisplayName("4. Generic Assessment endpoint /api/assessments/patient/{id} strictly isolates patient assessments")
    void testGenericAssessmentIsolation() throws Exception {
        mockMvc.perform(get("/api/assessments/patient/" + patientAId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(1)))
                .andExpect(jsonPath("$[0].id").value(assessmentAId))
                .andExpect(jsonPath("$[0].riskScore").value(0.1234));

        mockMvc.perform(get("/api/assessments/patient/" + patientBId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(1)))
                .andExpect(jsonPath("$[0].id").value(assessmentBId))
                .andExpect(jsonPath("$[0].riskScore").value(0.8765));
    }
}
