import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app, session_manager, extraction_engine
from app.llm_client import MockLLMClient

client = TestClient(app)

ASSESSMENT_PATIENT_A = {
    "id": 1011,
    "patientId": 101,
    "diseaseType": "HEART_DISEASE",
    "riskScore": 0.145,
    "riskLevel": "LOW",
    "assessedAt": "2026-10-02T10:00:00",
    "inputFeatures": json.dumps({
        "probability": 0.145,
        "prediction": 0,
        "threshold": 0.31,
        "modelName": "Logistic Regression (Robust L1)",
        "modelVersion": "1.0.0",
        "configId": "LR_robust_l1_C1.0",
        "explanationAvailable": True,
        "shapBaseValue": -0.0879,
        "positiveContributors": [
            {
                "feature": "age",
                "transformedFeature": "num__age",
                "displayName": "Age",
                "rawValue": 45,
                "contribution": 0.12,
                "absContribution": 0.12,
                "direction": "INCREASES_RISK_ESTIMATE",
            }
        ],
        "negativeContributors": [],
        "inputSummary": {"age": 45, "chol": 190, "isolationMarker": "PATIENT_A_EXCLUSIVE_MARKER"},
    }),
}

ASSESSMENT_PATIENT_B = {
    "id": 2022,
    "patientId": 202,
    "diseaseType": "HEART_DISEASE",
    "riskScore": 0.785,
    "riskLevel": "HIGH",
    "assessedAt": "2026-10-02T11:00:00",
    "inputFeatures": json.dumps({
        "probability": 0.785,
        "prediction": 1,
        "threshold": 0.31,
        "modelName": "Logistic Regression (Robust L1)",
        "modelVersion": "1.0.0",
        "configId": "LR_robust_l1_C1.0",
        "explanationAvailable": True,
        "shapBaseValue": -0.0879,
        "positiveContributors": [
            {
                "feature": "chol",
                "transformedFeature": "num__chol",
                "displayName": "Serum Cholesterol",
                "rawValue": 290,
                "contribution": 0.65,
                "absContribution": 0.65,
                "direction": "INCREASES_RISK_ESTIMATE",
            }
        ],
        "negativeContributors": [],
        "inputSummary": {"age": 68, "chol": 290, "isolationMarker": "PATIENT_B_EXCLUSIVE_MARKER"},
    }),
}


@pytest.fixture(autouse=True)
def reset_state():
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()


def test_heart_history_two_patient_isolation():
    """
    Verifies that when Patient A queries their Heart assessment history,
    the chatbot queries the backend with Patient A's patientId and returns only A's data,
    while Patient B queries with Patient B's patientId and returns only B's data,
    with zero cross-patient data leakage.
    """
    with patch("app.main.get_backend_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_patient_heart_assessments.side_effect = lambda pid: (
            [ASSESSMENT_PATIENT_A] if pid == 101 else ([ASSESSMENT_PATIENT_B] if pid == 202 else [])
        )
        mock_get_client.return_value = mock_client

        # Patient A queries past heart assessment
        res_a = client.post("/api/chat", json={
            "user_id": "101",
            "session_id": "session-a-01",
            "message": "What was my last heart assessment?"
        })
        assert res_a.status_code == 200
        data_a = res_a.json()
        msg_a = data_a["assistant_message"]

        # Verify backend client received Patient A's ID
        mock_client.get_patient_heart_assessments.assert_called_with(101)

        # Patient A receives Patient A's assessment output
        assert "14.5%" in msg_a or "0.145" in msg_a or "Low" in msg_a
        # Patient A MUST NOT receive Patient B's assessment data
        assert "78.5%" not in msg_a
        assert "2022" not in msg_a

        # Patient B queries past heart assessment
        res_b = client.post("/api/chat", json={
            "user_id": "202",
            "session_id": "session-b-01",
            "message": "What was my last heart assessment?"
        })
        assert res_b.status_code == 200
        data_b = res_b.json()
        msg_b = data_b["assistant_message"]

        # Verify backend client received Patient B's ID
        mock_client.get_patient_heart_assessments.assert_called_with(202)

        # Patient B receives Patient B's assessment output
        assert "78.5%" in msg_b or "0.785" in msg_b or "High" in msg_b
        # Patient B MUST NOT receive Patient A's assessment data
        assert "14.5%" not in msg_b
        assert "1011" not in msg_b
