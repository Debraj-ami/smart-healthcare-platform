package com.smarthealth.platform;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.smarthealth.platform.dto.HealthObservationRequest;
import com.smarthealth.platform.dto.TemporalContext;
import com.smarthealth.platform.dto.UserCreateRequest;
import com.smarthealth.platform.dto.UserResponse;
import com.smarthealth.platform.entity.ActivityLevel;
import com.smarthealth.platform.entity.HealthObservation;
import com.smarthealth.platform.entity.TimeOfDay;
import com.smarthealth.platform.entity.User;
import com.smarthealth.platform.entity.UserRole;
import com.smarthealth.platform.repository.HealthObservationRepository;
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
import java.time.LocalDateTime;
import java.time.LocalTime;

import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class ContextAwareTrendTests {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private UserService userService;

    @Autowired
    private ObservationService observationService;

    @Autowired
    private HealthObservationRepository observationRepository;

    private Long testPatientId;

    @BeforeEach
    void setUp() {
        UserResponse user = userService.createUser(new UserCreateRequest(
                "context." + System.nanoTime() + "@example.com", "password123", "Context", "User",
                LocalDate.of(1985, 4, 12), "Female", UserRole.PATIENT
        ));
        testPatientId = user.getId();
    }

    // --- TEMPORAL CONTEXT TESTS ---

    @Test
    void shouldCorrectlyClassifyTimeOfDayBoundaries() {
        LocalDate date = LocalDate.of(2026, 9, 23);

        // 00:00 -> NIGHT
        TemporalContext ctx00 = TemporalContext.fromDateTime(LocalDateTime.of(date, LocalTime.of(0, 0)));
        assertEquals(TimeOfDay.NIGHT, ctx00.getTimeOfDay());
        assertEquals(0, ctx00.getHourOfDay());

        // 05:59 -> NIGHT
        TemporalContext ctx0559 = TemporalContext.fromDateTime(LocalDateTime.of(date, LocalTime.of(5, 59)));
        assertEquals(TimeOfDay.NIGHT, ctx0559.getTimeOfDay());
        assertEquals(5, ctx0559.getHourOfDay());

        // 06:00 -> MORNING
        TemporalContext ctx0600 = TemporalContext.fromDateTime(LocalDateTime.of(date, LocalTime.of(6, 0)));
        assertEquals(TimeOfDay.MORNING, ctx0600.getTimeOfDay());
        assertEquals(6, ctx0600.getHourOfDay());

        // 11:59 -> MORNING
        TemporalContext ctx1159 = TemporalContext.fromDateTime(LocalDateTime.of(date, LocalTime.of(11, 59)));
        assertEquals(TimeOfDay.MORNING, ctx1159.getTimeOfDay());
        assertEquals(11, ctx1159.getHourOfDay());

        // 12:00 -> AFTERNOON
        TemporalContext ctx1200 = TemporalContext.fromDateTime(LocalDateTime.of(date, LocalTime.of(12, 0)));
        assertEquals(TimeOfDay.AFTERNOON, ctx1200.getTimeOfDay());
        assertEquals(12, ctx1200.getHourOfDay());

        // 17:59 -> AFTERNOON
        TemporalContext ctx1759 = TemporalContext.fromDateTime(LocalDateTime.of(date, LocalTime.of(17, 59)));
        assertEquals(TimeOfDay.AFTERNOON, ctx1759.getTimeOfDay());
        assertEquals(17, ctx1759.getHourOfDay());

        // 18:00 -> EVENING
        TemporalContext ctx1800 = TemporalContext.fromDateTime(LocalDateTime.of(date, LocalTime.of(18, 0)));
        assertEquals(TimeOfDay.EVENING, ctx1800.getTimeOfDay());
        assertEquals(18, ctx1800.getHourOfDay());

        // 23:59 -> EVENING
        TemporalContext ctx2359 = TemporalContext.fromDateTime(LocalDateTime.of(date, LocalTime.of(23, 59)));
        assertEquals(TimeOfDay.EVENING, ctx2359.getTimeOfDay());
        assertEquals(23, ctx2359.getHourOfDay());

        // Day of week
        assertEquals("WEDNESDAY", ctx1200.getDayOfWeek());
    }

    // --- HISTORICAL & BASELINE CONTEXT TESTS ---

    @Test
    void shouldComputeHistoricalAndRecentContextCorrectly() throws Exception {
        // Log 4 observations with 1 second spacing
        double[] values = {100.0, 110.0, 120.0, 150.0};
        for (double v : values) {
            observationService.recordObservation(new HealthObservationRequest(
                    testPatientId, "GLUCOSE", v, "mg/dL"
            ));
        }

        // Expected stats:
        // observationCount = 4
        // min = 100.0, max = 150.0
        // historicalMean = (100 + 110 + 120 + 150) / 4 = 120.0
        // recentMean (latest 3: 110, 120, 150) = 380 / 3 = 126.6667
        // diffFromHistoricalMean = 150.0 - 120.0 = 30.0
        // diffFromRecentMean = 150.0 - 126.6667 = 23.3333
        // previousValue = 120.0
        // valueChangeFromPrevious = 150.0 - 120.0 = 30.0
        // percentageChangeFromPrevious = (30.0 / 120.0) * 100 = 25.0%

        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentValue").value(150.0))
                .andExpect(jsonPath("$.previousValue").value(120.0))
                .andExpect(jsonPath("$.valueChangeFromPrevious").value(30.0))
                .andExpect(jsonPath("$.percentageChangeFromPrevious").value(closeTo(25.0, 0.001)))
                .andExpect(jsonPath("$.historicalContext.observationCount").value(4))
                .andExpect(jsonPath("$.historicalContext.historicalMinimum").value(100.0))
                .andExpect(jsonPath("$.historicalContext.historicalMaximum").value(150.0))
                .andExpect(jsonPath("$.historicalContext.historicalMean").value(closeTo(120.0, 0.001)))
                .andExpect(jsonPath("$.historicalContext.recentMean").value(closeTo(126.6667, 0.01)))
                .andExpect(jsonPath("$.historicalContext.differenceFromHistoricalMean").value(closeTo(30.0, 0.001)))
                .andExpect(jsonPath("$.historicalContext.differenceFromRecentMean").value(closeTo(23.3333, 0.01)))
                .andExpect(jsonPath("$.temporalContext.observationDate").isNotEmpty())
                .andExpect(jsonPath("$.temporalContext.timeOfDay").isNotEmpty())
                .andExpect(jsonPath("$.activityContext.activityLevel").value("UNKNOWN"));
    }

    @Test
    void shouldHandleFewerThanThreeObservationsForRecentMean() throws Exception {
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 110.0, "mg/dL"));
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 130.0, "mg/dL"));

        // Count = 2 (< 3) -> recentMean equals historicalMean = (110 + 130)/2 = 120.0
        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.historicalContext.observationCount").value(2))
                .andExpect(jsonPath("$.historicalContext.historicalMean").value(120.0))
                .andExpect(jsonPath("$.historicalContext.recentMean").value(120.0))
                .andExpect(jsonPath("$.previousValue").value(110.0))
                .andExpect(jsonPath("$.valueChangeFromPrevious").value(20.0));
    }

    @Test
    void shouldSafelyHandleZeroPreviousValueForPercentageChange() throws Exception {
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "METRIC_X", 0.0, "unit"));
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "METRIC_X", 50.0, "unit"));

        // previousValue = 0.0 -> percentage change must not be NaN or Infinity, handled as 0.0
        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/METRIC_X/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentValue").value(50.0))
                .andExpect(jsonPath("$.previousValue").value(0.0))
                .andExpect(jsonPath("$.valueChangeFromPrevious").value(50.0))
                .andExpect(jsonPath("$.percentageChangeFromPrevious").value(0.0));
    }

    // --- ACTIVITY CONTEXT TESTS ---

    @Test
    void shouldSaveAndExposeExplicitActivityContext() throws Exception {
        HealthObservationRequest req = new HealthObservationRequest(
                testPatientId, "GLUCOSE", 125.0, "mg/dL", ActivityLevel.MODERATE, 45
        );

        mockMvc.perform(post("/api/observations")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.activityLevel").value("MODERATE"))
                .andExpect(jsonPath("$.activityMinutes").value(45));

        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.activityContext.activityLevel").value("MODERATE"))
                .andExpect(jsonPath("$.activityContext.activityMinutes").value(45));
    }

    @Test
    void shouldDefaultToUnknownActivityWhenOmitted() throws Exception {
        HealthObservationRequest req = new HealthObservationRequest(
                testPatientId, "GLUCOSE", 112.0, "mg/dL"
        );

        mockMvc.perform(post("/api/observations")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.activityLevel").value("UNKNOWN"))
                .andExpect(jsonPath("$.activityMinutes").isEmpty());
    }

    // --- ISOLATION & SPECIFIC METRIC ENDPOINT TESTS ---

    @Test
    void shouldRetrieveTrendsForSpecificMetricAndPreserveIsolation() throws Exception {
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 100.0, "mg/dL"));
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "HEART_RATE", 80.0, "bpm"));

        // Analyze GLUCOSE
        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk());

        // GET /api/trends/patient/{id}/{metric}
        mockMvc.perform(get("/api/trends/patient/" + testPatientId + "/GLUCOSE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(1)))
                .andExpect(jsonPath("$[0].metricType").value("GLUCOSE"))
                .andExpect(jsonPath("$[0].historicalContext.observationCount").value(1));

        // GET HEART_RATE should be empty before analysis
        mockMvc.perform(get("/api/trends/patient/" + testPatientId + "/HEART_RATE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(0)));
    }

    @Test
    void shouldPreserveHistoricalSnapshotsWhenNewObservationsAreAddedLater() throws Exception {
        // 1. Initial 2 observations (no activity)
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 110.0, "mg/dL"));
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 120.0, "mg/dL"));

        // Analyze after obs 2 -> Analysis A
        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentValue").value(120.0))
                .andExpect(jsonPath("$.previousValue").value(110.0))
                .andExpect(jsonPath("$.valueChangeFromPrevious").value(10.0))
                .andExpect(jsonPath("$.trendDirection").value("INCREASING"))
                .andExpect(jsonPath("$.persistence").value(2))
                .andExpect(jsonPath("$.historicalContext.observationCount").value(2))
                .andExpect(jsonPath("$.activityContext.activityLevel").value("UNKNOWN"));

        // 2. Add obs 3 (145.0, no activity)
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 145.0, "mg/dL"));

        // Analyze after obs 3 -> Analysis B
        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentValue").value(145.0))
                .andExpect(jsonPath("$.previousValue").value(120.0))
                .andExpect(jsonPath("$.valueChangeFromPrevious").value(25.0))
                .andExpect(jsonPath("$.trendDirection").value("INCREASING"))
                .andExpect(jsonPath("$.persistence").value(3))
                .andExpect(jsonPath("$.historicalContext.observationCount").value(3))
                .andExpect(jsonPath("$.activityContext.activityLevel").value("UNKNOWN"));

        // 3. Add obs 4 (130.0, with RESTING activity)
        HealthObservationRequest req4 = new HealthObservationRequest(
                testPatientId, "GLUCOSE", 130.0, "mg/dL", ActivityLevel.RESTING, 0
        );
        observationService.recordObservation(req4);

        // Analyze after obs 4 -> Analysis C
        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentValue").value(130.0))
                .andExpect(jsonPath("$.previousValue").value(145.0))
                .andExpect(jsonPath("$.valueChangeFromPrevious").value(-15.0))
                .andExpect(jsonPath("$.trendDirection").value("DECREASING"))
                .andExpect(jsonPath("$.persistence").value(2))
                .andExpect(jsonPath("$.historicalContext.observationCount").value(4))
                .andExpect(jsonPath("$.activityContext.activityLevel").value("RESTING"))
                .andExpect(jsonPath("$.activityContext.activityMinutes").value(0));

        // 4. Retrieve all historical analyses (ordered by analyzedAt DESC: [C, B, A])
        mockMvc.perform(get("/api/trends/patient/" + testPatientId + "/GLUCOSE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(3)))
                // Analysis C (latest: index 0)
                .andExpect(jsonPath("$[0].currentValue").value(130.0))
                .andExpect(jsonPath("$[0].previousValue").value(145.0))
                .andExpect(jsonPath("$[0].valueChangeFromPrevious").value(-15.0))
                .andExpect(jsonPath("$[0].trendDirection").value("DECREASING"))
                .andExpect(jsonPath("$[0].historicalContext.observationCount").value(4))
                .andExpect(jsonPath("$[0].activityContext.activityLevel").value("RESTING"))
                // Analysis B (middle: index 1) - MUST PRESERVE ITS OWN SNAPSHOT
                .andExpect(jsonPath("$[1].currentValue").value(145.0))
                .andExpect(jsonPath("$[1].previousValue").value(120.0))
                .andExpect(jsonPath("$[1].valueChangeFromPrevious").value(25.0))
                .andExpect(jsonPath("$[1].trendDirection").value("INCREASING"))
                .andExpect(jsonPath("$[1].persistence").value(3))
                .andExpect(jsonPath("$[1].historicalContext.observationCount").value(3))
                .andExpect(jsonPath("$[1].activityContext.activityLevel").value("UNKNOWN"))
                // Analysis A (oldest: index 2) - MUST PRESERVE ITS OWN SNAPSHOT
                .andExpect(jsonPath("$[2].currentValue").value(120.0))
                .andExpect(jsonPath("$[2].previousValue").value(110.0))
                .andExpect(jsonPath("$[2].valueChangeFromPrevious").value(10.0))
                .andExpect(jsonPath("$[2].trendDirection").value("INCREASING"))
                .andExpect(jsonPath("$[2].persistence").value(2))
                .andExpect(jsonPath("$[2].historicalContext.observationCount").value(2))
                .andExpect(jsonPath("$[2].activityContext.activityLevel").value("UNKNOWN"));
    }

    @Test
    void shouldComputeAbsoluteAndSignedDeviationCorrectly() throws Exception {
        // Feed observations: 100 -> 120 -> 140 -> 130 -> 110
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 100.0, "mg/dL"));
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 120.0, "mg/dL"));
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 140.0, "mg/dL"));
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 130.0, "mg/dL"));

        // Baseline at obs 4: (100 + 120 + 140 + 130) / 4 = 122.5
        // Analyze at obs 4: current=130.0, baseline=122.5 -> absoluteDeviation=7.5, deviationFromBaseline=7.5
        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentValue").value(130.0))
                .andExpect(jsonPath("$.baselineValue").value(122.5))
                .andExpect(jsonPath("$.absoluteDeviation").value(7.5))
                .andExpect(jsonPath("$.deviationFromBaseline").value(7.5));

        // Add 5th observation: 110.0
        // Baseline at obs 5: (100 + 120 + 140 + 130 + 110) / 5 = 120.0
        observationService.recordObservation(new HealthObservationRequest(testPatientId, "GLUCOSE", 110.0, "mg/dL"));

        // Analyze at obs 5: current=110.0, baseline=120.0 -> absoluteDeviation=10.0, deviationFromBaseline=-10.0
        mockMvc.perform(post("/api/trends/patient/" + testPatientId + "/GLUCOSE/analyze"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.currentValue").value(110.0))
                .andExpect(jsonPath("$.baselineValue").value(120.0))
                .andExpect(jsonPath("$.absoluteDeviation").value(10.0))
                .andExpect(jsonPath("$.deviationFromBaseline").value(-10.0));
    }
}
