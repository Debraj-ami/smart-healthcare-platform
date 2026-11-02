package com.smarthealth.platform;

import com.smarthealth.platform.dto.UserCreateRequest;
import com.smarthealth.platform.dto.UserResponse;
import com.smarthealth.platform.entity.Alert;
import com.smarthealth.platform.entity.UserRole;
import com.smarthealth.platform.repository.AlertRepository;
import com.smarthealth.platform.service.UserService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDate;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class AlertControllerTests {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private UserService userService;

    @Autowired
    private AlertRepository alertRepository;

    @Test
    void shouldRetrieveAlertsForPatient() throws Exception {
        String email = "alert.patient." + System.currentTimeMillis() + "@example.com";
        UserResponse user = userService.createUser(new UserCreateRequest(
                email, "pass123", "Alert", "Patient",
                LocalDate.of(1985, 1, 1), "Male", UserRole.PATIENT
        ));

        Alert alert = new Alert(
                userService.getUserEntity(user.getId()),
                null,
                null,
                "ELEVATED_LIVER_RISK",
                "HIGH",
                "Liver assessment indicates high risk probability. Clinical review recommended.",
                "ACTIVE"
        );
        alertRepository.save(alert);

        mockMvc.perform(get("/api/alerts/patient/" + user.getId())
                        .accept(MediaType.APPLICATION_JSON))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray())
                .andExpect(jsonPath("$.length()").value(1))
                .andExpect(jsonPath("$[0].alertType").value("ELEVATED_LIVER_RISK"))
                .andExpect(jsonPath("$[0].severity").value("HIGH"))
                .andExpect(jsonPath("$[0].status").value("ACTIVE"));
    }
}
