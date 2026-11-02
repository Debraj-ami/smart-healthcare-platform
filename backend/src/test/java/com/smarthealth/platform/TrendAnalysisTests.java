package com.smarthealth.platform;

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
class TrendAnalysisTests {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private UserService userService;

    @Autowired
    private ObservationService observationService;

    private Long testPatientId;

    @BeforeEach
    void setUp() {
        UserResponse user = userService.createUser(new UserCreateRequest(
                "trend." + System.nanoTime() + "@example.com", "password123", "Trend", "User",
                LocalDate.of(1988, 5, 20), "Male", UserRole.PATIENT
        ));
        testPatientId = user.getId();
    }

    @Test
    void shouldReturn404WhenAnalyzingMetricWithNoObservations() throws Exception {
        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.message", containsString("No observations found")));
    }

    @Test
    void shouldAnalyzeSingleObservationAsStableWithPersistenceOneAndZeroDeviation() throws Exception {
        observationService.recordObservation(new HealthObservationRequest(
                testPatientId, "GLUCOSE", 120.0, "mg/dL"
        ));

        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.patientId").value(testPatientId))
                .andExpect(jsonPath("$.metricType").value("GLUCOSE"))
                .andExpect(jsonPath("$.currentValue").value(120.0))
                .andExpect(jsonPath("$.deviationScore").value(0.0))
                .andExpect(jsonPath("$.trendDirection").value("STABLE"))
                .andExpect(jsonPath("$.persistence").value(1))
                .andExpect(jsonPath("$.baselineId").isNumber());
    }

    @Test
    void shouldIdentifyIncreasingTrendAndComputePersistence() throws Exception {
        // Monotonically increasing sequence: 100, 105, 110, 115
        double[] values = {100.0, 105.0, 110.0, 115.0};
        for (double val : values) {
            observationService.recordObservation(new HealthObservationRequest(
                    testPatientId, "GLUCOSE", val, "mg/dL"
            ));
        }

        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentValue").value(115.0))
                .andExpect(jsonPath("$.trendDirection").value("INCREASING"))
                .andExpect(jsonPath("$.persistence").value(4));
    }

    @Test
    void shouldIdentifyDecreasingTrendAndComputePersistence() throws Exception {
        // Monotonically decreasing sequence: 140, 130, 120
        double[] values = {140.0, 130.0, 120.0};
        for (double val : values) {
            observationService.recordObservation(new HealthObservationRequest(
                    testPatientId, "HEART_RATE", val, "bpm"
            ));
        }

        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/HEART_RATE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentValue").value(120.0))
                .andExpect(jsonPath("$.trendDirection").value("DECREASING"))
                .andExpect(jsonPath("$.persistence").value(3));
    }

    @Test
    void shouldIdentifyStableTrendWhenRecentValuesAreEqual() throws Exception {
        double[] values = {105.0, 110.0, 110.0};
        for (double val : values) {
            observationService.recordObservation(new HealthObservationRequest(
                    testPatientId, "GLUCOSE", val, "mg/dL"
            ));
        }

        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentValue").value(110.0))
                .andExpect(jsonPath("$.trendDirection").value("STABLE"))
                .andExpect(jsonPath("$.persistence").value(2));
    }

    @Test
    void shouldComputeCorrectDeviationScoreAndRecentPersistenceAfterInflection() throws Exception {
        // Observations: 108, 110, 109, 112, 111, then 145
        // Baseline of 6 values: mean = 695 / 6 = 115.8333...
        // Latest two: 111 -> 145 is INCREASING. Previous was 112 -> 111 (decreasing).
        // Run of increasing from end is just (111 -> 145) = 2 observations.
        double[] values = {108.0, 110.0, 109.0, 112.0, 111.0, 145.0};
        for (double val : values) {
            observationService.recordObservation(new HealthObservationRequest(
                    testPatientId, "GLUCOSE", val, "mg/dL"
            ));
        }

        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentValue").value(145.0))
                .andExpect(jsonPath("$.trendDirection").value("INCREASING"))
                .andExpect(jsonPath("$.persistence").value(2))
                .andExpect(jsonPath("$.deviationScore").value(greaterThan(0.0)));
    }

    @Test
    void shouldPersistAndRetrieveHistoricalTrendsForPatient() throws Exception {
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 100.0, "mg/dL"));
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 105.0, "mg/dL"));

        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk());

        mockMvc.perform(get("/api/trends/patient/" + testPatientId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(1)))
                .andExpect(jsonPath("$[0].metricType").value("GLUCOSE"))
                .andExpect(jsonPath("$[0].trendDirection").value("INCREASING"));
    }

    @Test
    void shouldReturn404ForNonexistentPatient() throws Exception {
        mockMvc.perform(post("/api/trends/patient/999999/GLUCOSE/analyze"))
                .andExpect(status().isNotFound());

        mockMvc.perform(get("/api/trends/patient/999999"))
                .andExpect(status().isNotFound());
    }
}
