package com.smarthealth.platform;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.smarthealth.platform.dto.DiseaseAssessmentRequest;
import com.smarthealth.platform.dto.UserCreateRequest;
import com.smarthealth.platform.dto.UserResponse;
import com.smarthealth.platform.entity.DiseaseType;
import com.smarthealth.platform.entity.RiskLevel;
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
class AssessmentControllerTests {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private UserService userService;

    private Long testPatientId;

    @BeforeEach
    void setUp() {
        String email = "assess." + System.currentTimeMillis() + "@example.com";
        UserResponse user = userService.createUser(new UserCreateRequest(
                email, "password123", "Assess", "Patient",
                LocalDate.of(1975, 7, 10), "Male", UserRole.PATIENT
        ));
        testPatientId = user.getId();
    }

    @Test
    void shouldCreateAndRetrieveAssessment() throws Exception {
        DiseaseAssessmentRequest request = new DiseaseAssessmentRequest(
                testPatientId,
                DiseaseType.LIVER_DISEASE,
                0.78,
                RiskLevel.HIGH,
                "{\"total_bilirubin\": 2.3, \"direct_bilirubin\": 1.1, \"alkphos\": 280}"
        );

        mockMvc.perform(post("/api/assessments")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").isNumber())
                .andExpect(jsonPath("$.patientId").value(testPatientId))
                .andExpect(jsonPath("$.diseaseType").value("LIVER_DISEASE"))
                .andExpect(jsonPath("$.riskScore").value(0.78))
                .andExpect(jsonPath("$.riskLevel").value("HIGH"))
                .andExpect(jsonPath("$.inputFeatures").exists());

        mockMvc.perform(get("/api/assessments/patient/" + testPatientId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray())
                .andExpect(jsonPath("$[0].diseaseType").value("LIVER_DISEASE"));
    }
}
