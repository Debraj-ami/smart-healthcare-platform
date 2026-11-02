package com.smarthealth.platform;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.smarthealth.platform.dto.HealthObservationRequest;
import com.smarthealth.platform.dto.UserCreateRequest;
import com.smarthealth.platform.dto.UserResponse;
import com.smarthealth.platform.entity.UserRole;
import com.smarthealth.platform.service.ObservationService;
import com.smarthealth.platform.service.UserService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDate;

import static org.hamcrest.Matchers.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class PersonalBaselineTests {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private UserService userService;

    @Autowired
    private ObservationService observationService;

    private Long patientId1;
    private Long patientId2;

    @BeforeEach
    void setUp() {
        UserResponse user1 = userService.createUser(new UserCreateRequest(
                "baseline.p1." + System.nanoTime() + "@example.com", "password123", "P1", "Test",
                LocalDate.of(1990, 1, 1), "Male", UserRole.PATIENT
        ));
        patientId1 = user1.getId();

        UserResponse user2 = userService.createUser(new UserCreateRequest(
                "baseline.p2." + System.nanoTime() + "@example.com", "password123", "P2", "Test",
                LocalDate.of(1992, 2, 2), "Female", UserRole.PATIENT
        ));
        patientId2 = user2.getId();
    }

    @Test
    void shouldReturn404WhenNoObservationsExistForBaseline() throws Exception {
        mockMvc.perform(post("/api/baselines/patient/" + patientId1 + "/GLUCOSE/calculate"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.message", containsString("No observations found")));

        mockMvc.perform(get("/api/baselines/patient/" + patientId1 + "/GLUCOSE"))
                .andExpect(status().isNotFound());
    }

    @Test
    void shouldCalculateSingleObservationBaselineWithZeroVariability() throws Exception {
        observationService.recordObservation(new HealthObservationRequest(
                patientId1, "GLUCOSE", 110.0, "mg/dL"
        ));

        mockMvc.perform(get("/api/baselines/patient/" + patientId1 + "/GLUCOSE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.patientId").value(patientId1))
                .andExpect(jsonPath("$.metricType").value("GLUCOSE"))
                .andExpect(jsonPath("$.baselineValue").value(110.0))
                .andExpect(jsonPath("$.variability").value(0.0))
                .andExpect(jsonPath("$.observationCount").value(1));
    }

    @Test
    void shouldCalculateMultipleObservationsBaselineWithCorrectMeanAndPopulationStd() throws Exception {
        // Values: 108, 110, 109, 112, 111 -> Mean = 110.0, Population Std = sqrt(2.0) ~ 1.41421356
        double[] values = {108.0, 110.0, 109.0, 112.0, 111.0};
        for (double val : values) {
            observationService.recordObservation(new HealthObservationRequest(
                    patientId1, "GLUCOSE", val, "mg/dL"
            ));
        }

        mockMvc.perform(post("/api/baselines/patient/" + patientId1 + "/GLUCOSE/calculate"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.patientId").value(patientId1))
                .andExpect(jsonPath("$.metricType").value("GLUCOSE"))
                .andExpect(jsonPath("$.baselineValue").value(closeTo(110.0, 0.0001)))
                .andExpect(jsonPath("$.variability").value(closeTo(1.4142, 0.001)))
                .andExpect(jsonPath("$.observationCount").value(5));
    }

    @Test
    void shouldIncrementallyUpdateBaselineWhenNewObservationIsCreated() throws Exception {
        // 1. Initial 5 observations
        double[] initialValues = {108.0, 110.0, 109.0, 112.0, 111.0};
        for (double val : initialValues) {
            observationService.recordObservation(new HealthObservationRequest(
                    patientId1, "GLUCOSE", val, "mg/dL"
            ));
        }

        mockMvc.perform(get("/api/baselines/patient/" + patientId1 + "/GLUCOSE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.observationCount").value(5))
                .andExpect(jsonPath("$.baselineValue").value(closeTo(110.0, 0.0001)));

        // 2. Post new observation: 145.0 via REST endpoint
        HealthObservationRequest newObs = new HealthObservationRequest(
                patientId1, "GLUCOSE", 145.0, "mg/dL"
        );
        mockMvc.perform(post("/api/observations")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(newObs)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.metricValue").value(145.0));

        // 3. Verify baseline was automatically updated
        // New Mean = (550 + 145) / 6 = 115.8333...
        mockMvc.perform(get("/api/baselines/patient/" + patientId1 + "/GLUCOSE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.observationCount").value(6))
                .andExpect(jsonPath("$.baselineValue").value(closeTo(115.8333, 0.01)))
                .andExpect(jsonPath("$.variability").value(greaterThan(1.4142)));
    }

    @Test
    void shouldIsolateBaselinesBetweenPatients() throws Exception {
        // Patient 1 observations
        observationService.recordObservation(new HealthObservationRequest(patientId1, "GLUCOSE", 100.0, "mg/dL"));
        // Patient 2 observations
        observationService.recordObservation(new HealthObservationRequest(patientId2, "GLUCOSE", 160.0, "mg/dL"));

        mockMvc.perform(get("/api/baselines/patient/" + patientId1 + "/GLUCOSE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.baselineValue").value(100.0));

        mockMvc.perform(get("/api/baselines/patient/" + patientId2 + "/GLUCOSE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.baselineValue").value(160.0));
    }

    @Test
    void shouldIsolateBaselinesBetweenMetricsForSamePatient() throws Exception {
        observationService.recordObservation(new HealthObservationRequest(patientId1, "GLUCOSE", 100.0, "mg/dL"));
        observationService.recordObservation(new HealthObservationRequest(patientId1, "HEART_RATE", 75.0, "bpm"));

        mockMvc.perform(get("/api/baselines/patient/" + patientId1 + "/GLUCOSE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.metricType").value("GLUCOSE"))
                .andExpect(jsonPath("$.baselineValue").value(100.0));

        mockMvc.perform(get("/api/baselines/patient/" + patientId1 + "/HEART_RATE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.metricType").value("HEART_RATE"))
                .andExpect(jsonPath("$.baselineValue").value(75.0));

        mockMvc.perform(get("/api/baselines/patient/" + patientId1))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(2)));
    }

    @Test
    void shouldReturn404ForNonexistentPatient() throws Exception {
        mockMvc.perform(get("/api/baselines/patient/999999"))
                .andExpect(status().isNotFound());

        mockMvc.perform(get("/api/baselines/patient/999999/GLUCOSE"))
                .andExpect(status().isNotFound());
    }
}
