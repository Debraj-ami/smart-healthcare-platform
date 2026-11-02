import os
import logging
from typing import Dict, Any, Optional, List
import httpx

logger = logging.getLogger(__name__)


class BackendClientError(Exception):
    """Base exception for Spring Boot backend client errors."""
    pass


class BackendServiceUnavailableError(BackendClientError):
    """Raised when Spring Boot backend service is unreachable or returns 5xx."""
    pass


class BackendValidationError(BackendClientError):
    """Raised when Spring Boot backend returns 4xx client/validation error."""
    pass


class BackendClient:
    """
    HTTP client communicating with the Spring Boot backend service.
    
    The chatbot communicates with Spring Boot for:
      1. Clinical assessment predictions (POST /api/assessments/liver/predict)
      2. Returning-patient historical context (GET /api/trends, /api/baselines, /api/observations)
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 35.0):
        self.base_url = (
            base_url
            or os.getenv("BACKEND_BASE_URL")
            or os.getenv("SPRING_BOOT_URL")
            or "http://localhost:8080"
        ).rstrip("/")
        self.timeout = timeout

    def predict_liver(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls POST /api/assessments/liver/predict on Spring Boot.

        Args:
            payload: Dict containing the 10 required clinical parameters + patientId:
                patientId, age, gender, totalBilirubin, directBilirubin,
                alkalinePhosphatase, sgpt, sgot, totalProteins, albumin, agRatio

        Returns:
            Dict containing LiverPredictResponse:
                assessmentId, patientId, assessedAt, disease, probability,
                modelName, modelVersion, calibrationMethod, shapFeatures,
                inputSummary, disclaimer
        """
        url = f"{self.base_url}/api/assessments/liver/predict"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)

            if response.status_code in (200, 201):
                data = response.json()
                if not isinstance(data, dict):
                    raise BackendServiceUnavailableError("Invalid response structure from backend service.")
                return data

            if response.status_code == 400:
                logger.warning("Spring Boot validation rejected payload: %s", response.text[:200])
                raise BackendValidationError(f"Backend validation error (HTTP 400): {response.text[:200]}")

            if response.status_code == 503 or response.status_code >= 500:
                logger.warning("Spring Boot / ML service error (HTTP %d): %s", response.status_code, response.text[:200])
                raise BackendServiceUnavailableError(f"Backend service unavailable (HTTP {response.status_code}): {response.text[:200]}")

            raise BackendServiceUnavailableError(f"Unexpected backend response status {response.status_code}: {response.text[:200]}")

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException, httpx.NetworkError) as e:
            logger.warning("Failed to connect to backend at %s: %s", url, e)
            raise BackendServiceUnavailableError(f"Could not connect to backend service at {self.base_url}: {e}") from e
        except (BackendServiceUnavailableError, BackendValidationError):
            raise
        except Exception as e:
            logger.error("Unexpected error in BackendClient.predict_liver: %s", e, exc_info=True)
            raise BackendServiceUnavailableError(f"Unexpected backend communication error: {e}") from e

    def predict_heart(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls POST /api/assessments/heart/predict on Spring Boot.

        Args:
            payload: Dict containing the 13 required Cleveland clinical parameters + optional patientId:
                age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, patientId
                (ca and thal may be null/None)

        Returns:
            Dict containing HeartPredictResponse:
                assessmentId, patientId, assessedAt, disease, probability, prediction,
                threshold, thresholdDescription, modelName, modelVersion, configId,
                outputSpace, calibrationStatus, explanationAvailable, shapBaseValue,
                modelOutput, positiveContributors, negativeContributors, allContributions,
                inputSummary, disclaimer, patientContext
        """
        url = f"{self.base_url}/api/assessments/heart/predict"
        # Strict validation: never send target variable 'num'
        clean_payload = {k: v for k, v in payload.items() if k != "num"}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=clean_payload)

            if response.status_code in (200, 201):
                data = response.json()
                if not isinstance(data, dict):
                    raise BackendServiceUnavailableError("Invalid response structure from backend heart service.")
                return data

            if response.status_code == 400:
                logger.warning("Spring Boot validation rejected heart payload: %s", response.text[:200])
                raise BackendValidationError(f"Backend validation error (HTTP 400): {response.text[:200]}")

            if response.status_code == 503 or response.status_code >= 500:
                logger.warning("Spring Boot / ML service error (HTTP %d): %s", response.status_code, response.text[:200])
                raise BackendServiceUnavailableError(f"Backend service unavailable (HTTP {response.status_code}): {response.text[:200]}")

            raise BackendServiceUnavailableError(f"Unexpected backend response status {response.status_code}: {response.text[:200]}")

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException, httpx.NetworkError) as e:
            logger.warning("Failed to connect to backend heart prediction at %s: %s", url, e)
            raise BackendServiceUnavailableError(f"Could not connect to backend service at {self.base_url}: {e}") from e
        except (BackendServiceUnavailableError, BackendValidationError):
            raise
        except Exception as e:
            logger.error("Unexpected error in BackendClient.predict_heart: %s", e, exc_info=True)
            raise BackendServiceUnavailableError(f"Unexpected backend communication error: {e}") from e

    def get_patient_heart_assessments(self, patient_id: int) -> List[Dict[str, Any]]:
        """
        Retrieves real persisted historical Heart Disease assessments for an authenticated patient from Spring Boot.
        Calls GET /api/assessments/heart/patient/{patientId}.
        """
        url = f"{self.base_url}/api/assessments/heart/patient/{patient_id}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)

            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, list) else [data]
            if response.status_code == 404:
                return []
            if response.status_code >= 500 or response.status_code == 503:
                raise BackendServiceUnavailableError(f"Backend heart history service error (HTTP {response.status_code})")
            raise BackendServiceUnavailableError(f"Unexpected heart history response status {response.status_code}")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException, httpx.NetworkError) as e:
            logger.warning("Failed to connect to backend heart history at %s: %s", url, e)
            raise BackendServiceUnavailableError(f"Could not connect to backend service at {self.base_url}: {e}") from e
        except BackendServiceUnavailableError:
            raise
        except Exception as e:
            logger.error("Unexpected error in BackendClient.get_patient_heart_assessments: %s", e, exc_info=True)
            raise BackendServiceUnavailableError(f"Unexpected error retrieving patient heart assessments: {e}") from e

    def get_patient_heart_assessment_by_id(self, patient_id: int, assessment_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific real persisted Heart assessment by ID for an authenticated patient.
        Strictly enforces patient isolation by verifying the assessment belongs to patient_id.
        Returns None if not found or if the assessment does not belong to patient_id.
        """
        assessments = self.get_patient_heart_assessments(patient_id)
        for ass in assessments:
            if ass.get("id") == assessment_id or ass.get("assessmentId") == assessment_id:
                return ass
        return None

    def get_patient_trends(self, patient_id: int, metric_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves real persisted trend analysis snapshots for a patient from Spring Boot.
        Calls GET /api/trends/patient/{patientId} or /{patientId}/{metricType}.
        """
        if metric_type:
            url = f"{self.base_url}/api/trends/patient/{patient_id}/{metric_type.upper()}"
        else:
            url = f"{self.base_url}/api/trends/patient/{patient_id}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)

            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, list) else [data]
            if response.status_code == 404:
                return []
            if response.status_code >= 500 or response.status_code == 503:
                raise BackendServiceUnavailableError(f"Backend trends service error (HTTP {response.status_code})")
            raise BackendServiceUnavailableError(f"Unexpected trends response status {response.status_code}")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException, httpx.NetworkError) as e:
            logger.warning("Failed to connect to backend trends at %s: %s", url, e)
            raise BackendServiceUnavailableError(f"Could not connect to backend service at {self.base_url}: {e}") from e
        except BackendServiceUnavailableError:
            raise
        except Exception as e:
            logger.error("Unexpected error in BackendClient.get_patient_trends: %s", e, exc_info=True)
            raise BackendServiceUnavailableError(f"Unexpected error retrieving patient trends: {e}") from e

    def get_patient_baselines(self, patient_id: int, metric_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves computed personal baselines for a patient from Spring Boot.
        Calls GET /api/baselines/patient/{patientId} or /{patientId}/{metricType}.
        """
        if metric_type:
            url = f"{self.base_url}/api/baselines/patient/{patient_id}/{metric_type.upper()}"
        else:
            url = f"{self.base_url}/api/baselines/patient/{patient_id}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)

            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, list) else [data]
            if response.status_code == 404:
                return []
            if response.status_code >= 500 or response.status_code == 503:
                raise BackendServiceUnavailableError(f"Backend baselines service error (HTTP {response.status_code})")
            raise BackendServiceUnavailableError(f"Unexpected baselines response status {response.status_code}")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException, httpx.NetworkError) as e:
            logger.warning("Failed to connect to backend baselines at %s: %s", url, e)
            raise BackendServiceUnavailableError(f"Could not connect to backend service at {self.base_url}: {e}") from e
        except BackendServiceUnavailableError:
            raise
        except Exception as e:
            logger.error("Unexpected error in BackendClient.get_patient_baselines: %s", e, exc_info=True)
            raise BackendServiceUnavailableError(f"Unexpected error retrieving patient baselines: {e}") from e

    def get_patient_observations(self, patient_id: int) -> List[Dict[str, Any]]:
        """
        Retrieves historical health observations for a patient from Spring Boot.
        Calls GET /api/observations/patient/{patientId}.
        """
        url = f"{self.base_url}/api/observations/patient/{patient_id}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)

            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, list) else [data]
            if response.status_code == 404:
                return []
            if response.status_code >= 500 or response.status_code == 503:
                raise BackendServiceUnavailableError(f"Backend observations service error (HTTP {response.status_code})")
            raise BackendServiceUnavailableError(f"Unexpected observations response status {response.status_code}")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException, httpx.NetworkError) as e:
            logger.warning("Failed to connect to backend observations at %s: %s", url, e)
            raise BackendServiceUnavailableError(f"Could not connect to backend service at {self.base_url}: {e}") from e
        except BackendServiceUnavailableError:
            raise
        except Exception as e:
            logger.error("Unexpected error in BackendClient.get_patient_observations: %s", e, exc_info=True)
            raise BackendServiceUnavailableError(f"Unexpected error retrieving patient observations: {e}") from e

    def get_patient_health_context(self, patient_id: int, metric_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves and aggregates real historical patient context (trends, baselines, observations).
        Does not fabricate any data; returns has_history=False when no records exist.
        """
        trends = self.get_patient_trends(patient_id, metric_type=metric_type)
        baselines = self.get_patient_baselines(patient_id, metric_type=metric_type)
        observations = self.get_patient_observations(patient_id)

        has_data = bool(trends or baselines or observations)

        return {
            "patient_id": patient_id,
            "has_history": has_data,
            "trends": trends,
            "baselines": baselines,
            "recent_observations": observations[-10:] if observations else []
        }


_backend_client_instance: Optional[BackendClient] = None


def get_backend_client() -> BackendClient:
    """Provides singleton BackendClient instance or mock if injected."""
    global _backend_client_instance
    if _backend_client_instance is None:
        _backend_client_instance = BackendClient()
    return _backend_client_instance


def set_backend_client(client: Optional[BackendClient]) -> None:
    """Overrides backend client for automated testing."""
    global _backend_client_instance
    _backend_client_instance = client
