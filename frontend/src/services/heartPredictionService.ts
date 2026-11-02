import apiClient from './api';
import { DiseaseAssessment } from '../types/assessment';

/**
 * Types for the Heart Disease Prediction & Assessment API.
 * These mirror the Spring Boot HeartPredictResponse and HeartPredictRequest DTOs.
 */
export interface HeartFeatureContribution {
  feature: string;
  transformedFeature: string;
  displayName: string;
  rawValue: unknown;
  contribution: number;
  absContribution: number;
  direction: 'INCREASES_RISK_ESTIMATE' | 'DECREASES_RISK_ESTIMATE' | string;
  isCategorical?: boolean;
  isActive?: boolean;
  transformedValue?: number;
  displayStatus?: string;
}

export interface HeartPatientContext {
  patientId: number;
  gender: string;
  ageYears: number | null;
  previousAssessmentCount: number;
  lastAssessedAt: string | null;
  lastProbability: number | null;
  hasEstablishedBaselines: boolean;
  establishedBaselineMetrics: string[];
  contextNotes: string;
}

export interface HeartPredictResponse {
  assessmentId?: number | null;
  patientId?: number | null;
  assessedAt?: string | null;
  patientContext?: HeartPatientContext | null;
  disease: string;
  probability: number;
  prediction: number;
  threshold: number;
  thresholdDescription: string;
  modelName: string;
  modelVersion: string;
  configId: string;
  outputSpace: string;
  calibrationStatus: string;
  explanationAvailable: boolean;
  explanationOutputSpace: string;
  shapBaseValue: number;
  modelOutput: number;
  positiveContributors: HeartFeatureContribution[];
  negativeContributors: HeartFeatureContribution[];
  allContributions: HeartFeatureContribution[];
  explanationError?: string | null;
  inputSummary: Record<string, unknown>;
  disclaimer: string;
}

export interface HeartPredictRequest {
  patientId?: number;
  age: number;
  sex: number;
  cp: number;
  trestbps: number;
  chol: number;
  fbs: number;
  restecg: number;
  thalach: number;
  exang: number;
  oldpeak: number;
  slope: number;
  ca: number | null;
  thal: number | null;
}

/**
 * Service orchestrating Heart Disease prediction requests through Spring Boot.
 *
 * Flow:
 * React -> POST /api/assessments/heart/predict (Spring Boot)
 *       -> POST /api/predict/heart           (Python FastAPI ML microservice)
 *       -> Persists to disease_assessments
 *       -> Returns full prediction + SHAP explanation + patient context
 */
export const heartPredictionService = {
  /**
   * Executes a Heart Disease risk assessment.
   *
   * @param request Validated 13 Heart predictor inputs (+ optional patientId)
   * @returns Prediction response containing probability, prediction, threshold 0.31, SHAP attributions, and patient context
   */
  predict: async (request: HeartPredictRequest): Promise<HeartPredictResponse> => {
    const response = await apiClient.post<HeartPredictResponse>(
      '/api/assessments/heart/predict',
      request,
      { timeout: 35000 }
    );
    return response.data;
  },

  /**
   * Retrieves historical Heart Disease assessments for an authenticated patient.
   *
   * @param patientId ID of the patient
   * @returns Array of recorded Heart disease assessments in reverse chronological order
   */
  getPatientHeartHistory: async (patientId: number): Promise<DiseaseAssessment[]> => {
    const response = await apiClient.get<DiseaseAssessment[]>(
      `/api/assessments/heart/patient/${patientId}`
    );
    return response.data;
  },
};

export default heartPredictionService;
