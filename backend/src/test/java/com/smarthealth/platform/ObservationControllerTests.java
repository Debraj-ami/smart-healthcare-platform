package com.smarthealth.platform;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.smarthealth.platform.dto.HealthObservationRequest;
import com.smarthealth.platform.dto.UserCreateRequest;
import com.smarthealth.platform.dto.UserResponse;
import com.smarthealth.platform.entity.UserRole;
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

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class ObservationControllerTests {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private UserService userService;

    private Long testPatientId;

    @BeforeEach
    void setUp() {
        String email = "obs." + System.currentTimeMillis() + "@example.com";
        UserResponse user = userService.createUser(new UserCreateRequest(
                email, "password123", "Obs", "User",
                LocalDate.of(1982, 3, 15), "Female", UserRole.PATIENT
        ));
        testPatientId = user.getId();
    }

    @Test
    void shouldCreateAndRetrieveObservation() throws Exception {
        HealthObservationRequest request = new HealthObservationRequest(
                testPatientId, "blood_glucose", 105.0, "mg/dL"
        );

        mockMvc.perform(post("/api/observations")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").isNumber())
                .andExpect(jsonPath("$.patientId").value(testPatientId))
                .andExpect(jsonPath("$.metricType").value("blood_glucose"))
                .andExpect(jsonPath("$.metricValue").value(105.0))
                .andExpect(jsonPath("$.unit").value("mg/dL"));

        mockMvc.perform(get("/api/observations/patient/" + testPatientId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray())
                .andExpect(jsonPath("$[0].metricType").value("blood_glucose"));
    }
}
