package com.smarthealth.platform;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.smarthealth.platform.dto.HeartPredictRequest;
import com.smarthealth.platform.dto.UserCreateRequest;
import com.smarthealth.platform.dto.UserResponse;
import com.smarthealth.platform.entity.DiseaseAssessment;
import com.smarthealth.platform.entity.DiseaseType;
import com.smarthealth.platform.entity.PersonalBaseline;
import com.smarthealth.platform.entity.UserRole;
import com.smarthealth.platform.repository.DiseaseAssessmentRepository;
import com.smarthealth.platform.repository.PersonalBaselineRepository;
import com.smarthealth.platform.service.UserService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.client.ExpectedCount;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.test.web.client.match.MockRestRequestMatchers;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.web.client.RestTemplate;

import java.net.SocketTimeoutException;
import java.time.LocalDate;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.client.response.MockRestResponseCreators.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Integration and contract tests for Heart Disease prediction orchestration in Spring Boot.
 *
 * Mocks at the HTTP network boundary (RestTemplate / MockRestServiceServer) to test:
 * 1. Valid Heart request accepts all 13 predictor fields.
 * 2. All 13 fields are serialized and forwarded to ML service correctly.
 * 3. Target variable 'num' is strictly rejected and never forwarded.
 * 4. Nullable fields ca=null and thal=null succeed per Phase 4 contract.
 * 5. ML service response is mapped completely (probability, prediction, threshold 0.31, model metadata, SHAP).
 * 6. Isolated SHAP failure (explanation.available=false) is preserved without fabricating numbers.
 * 7. ML service unavailability (503) is handled gracefully.
 * 8. ML service error responses (500) are handled safely.
 * 9. Malformed ML responses (missing probability) are rejected without fabricating fallbacks.
 * 10. Invalid categorical values, out-of-range bounds, and missing required fields are rejected (400).
 * 11. Patient-context persistence to disease_assessments table works when patientId is provided.
 * 12. Patient context enrichment: previous assessments, baseline status, and stateless execution.
 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class HeartPredictControllerTests {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private UserService userService;

    @Autowired
    private DiseaseAssessmentRepository assessmentRepository;

    @Autowired
    private PersonalBaselineRepository personalBaselineRepository;

    @Autowired
    private RestTemplate restTemplate;

    private MockRestServiceServer mockServer;

    private Long testPatientId;

    private static final String FASTAPI_HEART_URL = "http://localhost:8001/api/predict/heart";

    private static final String SAMPLE_SUCCESS_ML_JSON = """
        {
          "disease": "HEART_DISEASE",
          "model": {
            "name": "Logistic Regression (Robust L1)",
            "version": "1.0.0",
            "config_id": "LR_robust_l1_C1.0",
            "output_space": "log_odds",
            "calibration_status": "Uncalibrated Baseline (Empirically Calibrated)"
          },
          "probability": 0.8872,
          "prediction": 1,
          "threshold": 0.31,
          "threshold_description": "Configured model classification threshold (t=0.31) targeting >= 90% sensitivity established in Phase 6.",
          "explanation": {
            "available": true,
            "output_space": "log_odds",
            "base_value": -0.2514,
            "model_output": 2.0625,
            "positive_contributors": [
              {
                "feature": "thal",
                "transformed_feature": "cat__thal_7.0",
                "display_name": "Thallium: Reversible Defect",
                "raw_value": 7.0,
                "contribution": 0.9421,
                "abs_contribution": 0.9421,
                "direction": "INCREASES_RISK_ESTIMATE"
              }
            ],
            "negative_contributors": [
              {
                "feature": "thalach",
                "transformed_feature": "num__thalach",
                "display_name": "Maximum Heart Rate",
                "raw_value": 112.0,
                "contribution": -0.3142,
                "abs_contribution": 0.3142,
                "direction": "DECREASES_RISK_ESTIMATE"
              }
            ],
            "all_contributions": [
              {
                "feature": "thal",
                "transformed_feature": "cat__thal_7.0",
                "display_name": "Thallium: Reversible Defect",
                "raw_value": 7.0,
                "contribution": 0.9421,
                "abs_contribution": 0.9421,
                "direction": "INCREASES_RISK_ESTIMATE"
              },
              {
                "feature": "thalach",
                "transformed_feature": "num__thalach",
                "display_name": "Maximum Heart Rate",
                "raw_value": 112.0,
                "contribution": -0.3142,
                "abs_contribution": 0.3142,
                "direction": "DECREASES_RISK_ESTIMATE"
              }
            ],
            "error": null
          },
          "input_summary": {
            "age": 63.0,
            "sex": 1.0,
            "cp": 4.0
          },
          "disclaimer": "Research decision support screening tool."
        }
        """;

    @BeforeEach
    void setUp() {
        mockServer = MockRestServiceServer.bindTo(restTemplate).ignoreExpectOrder(true).build();

        String email = "heart.test." + System.currentTimeMillis() + "@example.com";
        UserResponse user = userService.createUser(new UserCreateRequest(
                email, "password123", "Heart", "Patient",
                LocalDate.of(1980, 5, 12), "Male", UserRole.PATIENT
        ));
        testPatientId = user.getId();
    }

    private HeartPredictRequest createValidRequest() {
        HeartPredictRequest req = new HeartPredictRequest();
        req.setAge(63.0);
        req.setSex(1.0);
        req.setCp(4.0);
        req.setTrestbps(140.0);
        req.setChol(260.0);
        req.setFbs(0.0);
        req.setRestecg(1.0);
        req.setThalach(112.0);
        req.setExang(1.0);
        req.setOldpeak(3.0);
        req.setSlope(2.0);
        req.setCa(2.0);
        req.setThal(7.0);
        return req;
    }

    // =====================================================================
    // 1. HAPPY PATH & FIELD FORWARDING TESTS
    // =====================================================================

    @Test
    @DisplayName("Should accept valid Heart request and return full mapped prediction")
    void shouldAcceptValidHeartRequestAndReturnFullPrediction() throws Exception {
        mockServer.expect(ExpectedCount.once(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andExpect(MockRestRequestMatchers.method(HttpMethod.POST))
                .andRespond(withSuccess(SAMPLE_SUCCESS_ML_JSON, MediaType.APPLICATION_JSON));

        HeartPredictRequest request = createValidRequest();

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.disease").value("HEART_DISEASE"))
                .andExpect(jsonPath("$.probability").value(0.8872))
                .andExpect(jsonPath("$.prediction").value(1))
                .andExpect(jsonPath("$.threshold").value(0.31))
                .andExpect(jsonPath("$.thresholdDescription").exists())
                .andExpect(jsonPath("$.modelName").value("Logistic Regression (Robust L1)"))
                .andExpect(jsonPath("$.modelVersion").value("1.0.0"))
                .andExpect(jsonPath("$.configId").value("LR_robust_l1_C1.0"))
                .andExpect(jsonPath("$.explanationAvailable").value(true))
                .andExpect(jsonPath("$.explanationOutputSpace").value("log_odds"))
                .andExpect(jsonPath("$.shapBaseValue").value(-0.2514))
                .andExpect(jsonPath("$.modelOutput").value(2.0625))
                .andExpect(jsonPath("$.positiveContributors[0].feature").value("thal"))
                .andExpect(jsonPath("$.positiveContributors[0].transformedFeature").value("cat__thal_7.0"))
                .andExpect(jsonPath("$.positiveContributors[0].displayName").value("Thallium: Reversible Defect"))
                .andExpect(jsonPath("$.positiveContributors[0].contribution").value(0.9421))
                .andExpect(jsonPath("$.positiveContributors[0].direction").value("INCREASES_RISK_ESTIMATE"))
                .andExpect(jsonPath("$.negativeContributors[0].feature").value("thalach"))
                .andExpect(jsonPath("$.negativeContributors[0].contribution").value(-0.3142))
                .andExpect(jsonPath("$.negativeContributors[0].direction").value("DECREASES_RISK_ESTIMATE"))
                .andExpect(jsonPath("$.allContributions").isArray())
                .andExpect(jsonPath("$.disclaimer").exists());

        mockServer.verify();
    }

    @Test
    @DisplayName("Should verify all 13 predictor fields are serialized to the ML service")
    void shouldVerifyAll13FieldsForwardedToMlService() throws Exception {
        mockServer.expect(ExpectedCount.once(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andExpect(MockRestRequestMatchers.method(HttpMethod.POST))
                .andExpect(MockRestRequestMatchers.header("Content-Type", MediaType.APPLICATION_JSON_VALUE))
                .andExpect(MockRestRequestMatchers.jsonPath("$.age").value(63.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.sex").value(1.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.cp").value(4.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.trestbps").value(140.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.chol").value(260.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.fbs").value(0.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.restecg").value(1.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.thalach").value(112.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.exang").value(1.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.oldpeak").value(3.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.slope").value(2.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.ca").value(2.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.thal").value(7.0))
                .andExpect(MockRestRequestMatchers.jsonPath("$.num").doesNotExist()) // Never forward target variable
                .andRespond(withSuccess(SAMPLE_SUCCESS_ML_JSON, MediaType.APPLICATION_JSON));

        HeartPredictRequest request = createValidRequest();

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated());

        mockServer.verify();
    }

    // =====================================================================
    // 2. TARGET VARIABLE 'num' REJECTION
    // =====================================================================

    @Test
    @DisplayName("Should strictly reject target variable 'num' with HTTP 400")
    void shouldStrictlyRejectTargetVariableNum() throws Exception {
        String payloadWithNum = """
            {
                "age": 63.0,
                "sex": 1.0,
                "cp": 4.0,
                "trestbps": 140.0,
                "chol": 260.0,
                "fbs": 0.0,
                "restecg": 1.0,
                "thalach": 112.0,
                "exang": 1.0,
                "oldpeak": 3.0,
                "slope": 2.0,
                "ca": 2.0,
                "thal": 7.0,
                "num": 1
            }
            """;

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(payloadWithNum))
                .andExpect(status().isBadRequest());

        // Verify ML service is never called when num is provided
        mockServer.verify();
    }

    // =====================================================================
    // 3. NULLABLE ca AND thal PER PHASE 4 CONTRACT
    // =====================================================================

    @Test
    @DisplayName("Should accept nullable ca=null and thal=null per Phase 4 contract")
    void shouldAcceptNullableCaAndThal() throws Exception {
        mockServer.expect(ExpectedCount.once(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andExpect(MockRestRequestMatchers.method(HttpMethod.POST))
                .andExpect(MockRestRequestMatchers.jsonPath("$.ca").doesNotExist())
                .andExpect(MockRestRequestMatchers.jsonPath("$.thal").doesNotExist())
                .andRespond(withSuccess(SAMPLE_SUCCESS_ML_JSON, MediaType.APPLICATION_JSON));

        HeartPredictRequest request = createValidRequest();
        request.setCa(null);
        request.setThal(null);

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated());

        mockServer.verify();
    }

    // =====================================================================
    // 4. SHAP EXPLANATION FAILURE ISOLATION
    // =====================================================================

    @Test
    @DisplayName("Should preserve explanation.available=false without fabricating values")
    void shouldPreserveExplanationUnavailableWithoutFabricatingValues() throws Exception {
        String failureShapJson = """
            {
              "disease": "HEART_DISEASE",
              "model": {
                "name": "Logistic Regression (Robust L1)",
                "version": "1.0.0",
                "config_id": "LR_robust_l1_C1.0",
                "output_space": "log_odds",
                "calibration_status": "Uncalibrated Baseline (Empirically Calibrated)"
              },
              "probability": 0.8872,
              "prediction": 1,
              "threshold": 0.31,
              "threshold_description": "Configured threshold 0.31",
              "explanation": {
                "available": false,
                "output_space": "log_odds",
                "base_value": 0.0,
                "model_output": 2.0625,
                "positive_contributors": [],
                "negative_contributors": [],
                "all_contributions": [],
                "error": "SHAP explanation failed: runtime background dimension mismatch"
              },
              "input_summary": {},
              "disclaimer": "Research tool."
            }
            """;

        mockServer.expect(ExpectedCount.once(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andRespond(withSuccess(failureShapJson, MediaType.APPLICATION_JSON));

        HeartPredictRequest request = createValidRequest();

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.probability").value(0.8872))
                .andExpect(jsonPath("$.prediction").value(1))
                .andExpect(jsonPath("$.explanationAvailable").value(false))
                .andExpect(jsonPath("$.explanationError").value("SHAP explanation failed: runtime background dimension mismatch"))
                .andExpect(jsonPath("$.positiveContributors").isEmpty())
                .andExpect(jsonPath("$.negativeContributors").isEmpty())
                .andExpect(jsonPath("$.allContributions").isEmpty());

        mockServer.verify();
    }

    // =====================================================================
    // 5. ERROR RESILIENCE & INTEGRATION FAILURE
    // =====================================================================

    @Test
    @DisplayName("Should return 503 Service Unavailable when ML microservice is unreachable")
    void shouldReturn503WhenMlServiceUnreachable() throws Exception {
        mockServer.expect(ExpectedCount.once(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andRespond(withException(new SocketTimeoutException("Connection timed out to FastAPI")));

        HeartPredictRequest request = createValidRequest();

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.error").value("ML service unavailable"))
                .andExpect(jsonPath("$.message").exists())
                .andExpect(jsonPath("$.hint").exists());

        mockServer.verify();
    }

    @Test
    @DisplayName("Should return 503 when ML microservice returns HTTP 500 error")
    void shouldReturn503WhenMlServiceReturnsHttp500() throws Exception {
        mockServer.expect(ExpectedCount.once(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andRespond(withServerError());

        HeartPredictRequest request = createValidRequest();

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isServiceUnavailable());

        mockServer.verify();
    }

    @Test
    @DisplayName("Should reject malformed ML response missing probability without fabricating 0.0")
    void shouldRejectMalformedMlResponseMissingProbability() throws Exception {
        String malformedJson = """
            {
              "disease": "HEART_DISEASE",
              "probability": null
            }
            """;

        mockServer.expect(ExpectedCount.once(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andRespond(withSuccess(malformedJson, MediaType.APPLICATION_JSON));

        HeartPredictRequest request = createValidRequest();

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isServiceUnavailable());

        mockServer.verify();
    }

    // =====================================================================
    // 6. INPUT VALIDATION & BOUNDARY TESTING
    // =====================================================================

    @Test
    @DisplayName("Should reject request with missing required field (age)")
    void shouldRejectMissingRequiredField() throws Exception {
        HeartPredictRequest request = createValidRequest();
        request.setAge(null);

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.details.age").exists());


        mockServer.verify();
    }

    @Test
    @DisplayName("Should reject invalid categorical and discrete values")
    void shouldRejectInvalidCategoricalValues() throws Exception {
        // Bad sex (must be 0 or 1)
        HeartPredictRequest badSex = createValidRequest();
        badSex.setSex(2.0);
        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(badSex)))
                .andExpect(status().isBadRequest());

        // Bad cp (must be 1, 2, 3, 4)
        HeartPredictRequest badCp = createValidRequest();
        badCp.setCp(5.0);
        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(badCp)))
                .andExpect(status().isBadRequest());

        // Bad slope (must be 1, 2, 3)
        HeartPredictRequest badSlope = createValidRequest();
        badSlope.setSlope(0.0);
        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(badSlope)))
                .andExpect(status().isBadRequest());

        // Bad thal (must be null, 3, 6, 7)
        HeartPredictRequest badThal = createValidRequest();
        badThal.setThal(5.0);
        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(badThal)))
                .andExpect(status().isBadRequest());

        mockServer.verify();
    }

    @Test
    @DisplayName("Should reject out-of-range numerical values")
    void shouldRejectOutOfRangeNumericalValues() throws Exception {
        // Age > 120
        HeartPredictRequest badAge = createValidRequest();
        badAge.setAge(150.0);
        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(badAge)))
                .andExpect(status().isBadRequest());

        // Blood pressure < 50
        HeartPredictRequest badBp = createValidRequest();
        badBp.setTrestbps(20.0);
        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(badBp)))
                .andExpect(status().isBadRequest());

        mockServer.verify();
    }

    // =====================================================================
    // 7. PERSISTENCE IN PATIENT CONTEXT
    // =====================================================================

    @Test
    @DisplayName("Should persist assessment and populate patientContext when patientId is provided")
    void shouldPersistAssessmentWhenPatientIdProvided() throws Exception {
        mockServer.expect(ExpectedCount.once(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andRespond(withSuccess(SAMPLE_SUCCESS_ML_JSON, MediaType.APPLICATION_JSON));

        HeartPredictRequest request = createValidRequest();
        request.setPatientId(testPatientId);

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.assessmentId").isNumber())
                .andExpect(jsonPath("$.patientId").value(testPatientId))
                .andExpect(jsonPath("$.assessedAt").exists())
                .andExpect(jsonPath("$.patientContext").exists())
                .andExpect(jsonPath("$.patientContext.patientId").value(testPatientId))
                .andExpect(jsonPath("$.patientContext.gender").value("Male"))
                .andExpect(jsonPath("$.patientContext.previousAssessmentCount").value(0))
                .andExpect(jsonPath("$.patientContext.lastProbability").doesNotExist())
                .andExpect(jsonPath("$.patientContext.hasEstablishedBaselines").value(false))
                .andExpect(jsonPath("$.patientContext.establishedBaselineMetrics").isEmpty())
                .andExpect(jsonPath("$.patientContext.contextNotes").exists());

        // Verify entity persisted in disease_assessments table
        List<DiseaseAssessment> saved = assessmentRepository.findByPatientIdOrderByAssessedAtDesc(testPatientId);
        assertThat(saved).isNotEmpty();
        assertThat(saved.get(0).getDiseaseType()).isEqualTo(DiseaseType.HEART_DISEASE);
        assertThat(saved.get(0).getRiskScore()).isEqualTo(0.8872);
        assertThat(saved.get(0).getRiskLevel()).isNull(); // No arbitrary category forced
        assertThat(saved.get(0).getInputFeatures()).contains("shapBaseValue");

        mockServer.verify();
    }

    @Test
    @DisplayName("Should return null patientContext and no persistence when patientId is omitted")
    void shouldReturnNullPatientContextWhenPatientIdOmitted() throws Exception {
        mockServer.expect(ExpectedCount.once(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andRespond(withSuccess(SAMPLE_SUCCESS_ML_JSON, MediaType.APPLICATION_JSON));

        HeartPredictRequest request = createValidRequest();
        request.setPatientId(null);

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.probability").value(0.8872))
                .andExpect(jsonPath("$.prediction").value(1))
                .andExpect(jsonPath("$.threshold").value(0.31))
                .andExpect(jsonPath("$.assessmentId").doesNotExist())
                .andExpect(jsonPath("$.patientId").doesNotExist())
                .andExpect(jsonPath("$.patientContext").doesNotExist());

        mockServer.verify();
    }

    @Test
    @DisplayName("Should return 404 when patientId does not exist in the database")
    void shouldReturn404WhenPatientIdNotFound() throws Exception {
        HeartPredictRequest request = createValidRequest();
        request.setPatientId(999999L);

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404))
                .andExpect(jsonPath("$.message").value("User not found with id: 999999"));

        // Verify ML service is NOT called when patient resolution fails
        mockServer.verify();
    }

    @Test
    @DisplayName("Should retrieve previous Heart assessment metadata on subsequent assessment for same patient")
    void shouldRetrievePreviousHeartAssessmentMetadataOnSubsequentAssessment() throws Exception {
        // Expect two consecutive ML requests upfront
        mockServer.expect(ExpectedCount.twice(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andRespond(withSuccess(SAMPLE_SUCCESS_ML_JSON, MediaType.APPLICATION_JSON));

        HeartPredictRequest request1 = createValidRequest();
        request1.setPatientId(testPatientId);

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request1)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.patientContext.previousAssessmentCount").value(0));

        // Second assessment for the same patient
        HeartPredictRequest request2 = createValidRequest();
        request2.setPatientId(testPatientId);

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request2)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.patientContext.previousAssessmentCount").value(1))
                .andExpect(jsonPath("$.patientContext.lastProbability").value(0.8872))
                .andExpect(jsonPath("$.patientContext.lastAssessedAt").exists());

        mockServer.verify();
    }

    @Test
    @DisplayName("Should report established baselines when present and not fabricate unrecorded metrics")
    void shouldReportEstablishedBaselinesWhenPresent() throws Exception {
        com.smarthealth.platform.entity.User patient = userService.getUserEntity(testPatientId);
        personalBaselineRepository.save(new PersonalBaseline(
                patient, "BLOOD_PRESSURE_SYSTOLIC", 130.0, 5.2, 8
        ));

        mockServer.expect(ExpectedCount.once(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andRespond(withSuccess(SAMPLE_SUCCESS_ML_JSON, MediaType.APPLICATION_JSON));

        HeartPredictRequest request = createValidRequest();
        request.setPatientId(testPatientId);

        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.patientContext.hasEstablishedBaselines").value(true))
                .andExpect(jsonPath("$.patientContext.establishedBaselineMetrics[0]").value("BLOOD_PRESSURE_SYSTOLIC"));

        mockServer.verify();
    }

    @Test
    @DisplayName("Should retrieve historical Heart assessments via GET endpoint in reverse chronological order")
    void shouldRetrieveHistoricalHeartAssessmentsEndpoint() throws Exception {
        mockServer.expect(ExpectedCount.once(), MockRestRequestMatchers.requestTo(FASTAPI_HEART_URL))
                .andRespond(withSuccess(SAMPLE_SUCCESS_ML_JSON, MediaType.APPLICATION_JSON));

        HeartPredictRequest request = createValidRequest();
        request.setPatientId(testPatientId);

        // Perform one assessment
        mockMvc.perform(post("/api/assessments/heart/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated());

        // Query historical assessments
        mockMvc.perform(get("/api/assessments/heart/patient/" + testPatientId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray())
                .andExpect(jsonPath("$[0].patientId").value(testPatientId))
                .andExpect(jsonPath("$[0].diseaseType").value("HEART_DISEASE"))
                .andExpect(jsonPath("$[0].riskScore").value(0.8872));

        mockServer.verify();
    }

    @Test
    @DisplayName("Should return 404 when querying historical Heart assessments for non-existent patient")
    void shouldReturn404ForHistoricalAssessmentsWhenPatientNotFound() throws Exception {
        mockMvc.perform(get("/api/assessments/heart/patient/999999"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404));
    }
}
