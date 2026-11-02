package com.smarthealth.platform;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.smarthealth.platform.dto.PatientProfileRequest;
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
class PatientControllerTests {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private UserService userService;

    private Long testPatientId;

    @BeforeEach
    void setUp() {
        String email = "patient." + System.currentTimeMillis() + "@example.com";
        UserResponse user = userService.createUser(new UserCreateRequest(
                email, "password123", "Patient", "Test",
                LocalDate.of(1980, 1, 1), "Male", UserRole.PATIENT
        ));
        testPatientId = user.getId();
    }

    @Test
    void shouldReturnDefaultProfileWhenNotYetCreated() throws Exception {
        mockMvc.perform(get("/api/patients/" + testPatientId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.userId").value(testPatientId))
                .andExpect(jsonPath("$.patientId").value(testPatientId))
                .andExpect(jsonPath("$.bloodGroup").doesNotExist());
    }

    @Test
    void shouldReturn404WhenUserNotFound() throws Exception {
        mockMvc.perform(get("/api/patients/999999"))
                .andExpect(status().isNotFound());
    }

    @Test
    void shouldCreateAndRetrievePatientProfile() throws Exception {
        PatientProfileRequest profileRequest = new PatientProfileRequest(
                "O+", 175.5, 72.0, "Non-smoker, moderate exercise", "No known allergies"
        );

        mockMvc.perform(post("/api/patients/" + testPatientId + "/profile")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(profileRequest)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.userId").value(testPatientId))
                .andExpect(jsonPath("$.bloodGroup").value("O+"))
                .andExpect(jsonPath("$.heightCm").value(175.5))
                .andExpect(jsonPath("$.weightKg").value(72.0));

        mockMvc.perform(get("/api/patients/" + testPatientId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.userId").value(testPatientId))
                .andExpect(jsonPath("$.bloodGroup").value("O+"))
                .andExpect(jsonPath("$.lifestyleContext").value("Non-smoker, moderate exercise"));
    }
}
