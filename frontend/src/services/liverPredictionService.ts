import apiClient from './api';

/**
 * Types for the liver disease prediction API.
 * These mirror the Spring Boot LiverPredictResponse DTO.
 */
export interface FeatureContribution {
  feature: string;
  transformedFeature: string;
  value: number | string;
  contribution: number;
  direction: 'INCREASES_RISK' | 'DECREASES_RISK' | 'NEUTRAL';
}

export interface LiverPredictResponse {
  assessmentId: number;
  patientId: number;
  assessedAt: string;
  disease: string;
  probability: number;
  modelName: string;
  modelVersion: string;
  calibrationMethod: string;
  shapBaseValue: number;
  shapOutputSpace: string;
  shapFeatures: FeatureContribution[];
  inputSummary: Record<string, number | string>;
  disclaimer: string;
}

export interface LiverPredictRequest {
  patientId: number;
  age: number;
  gender: 'Male' | 'Female';
  totalBilirubin: number;
  directBilirubin: number;
  alkalinePhosphatase: number;
  sgpt: number;
  sgot: number;
  totalProteins: number;
  albumin: number;
  agRatio: number;
}

/**
 * Calls POST /api/assessments/liver/predict on the Spring Boot backend.
 * Spring Boot orchestrates the call to the Python ML service, persists the result,
 * and returns the full prediction + SHAP explanation.
 */
export const liverPredictionService = {
  predict: async (request: LiverPredictRequest): Promise<LiverPredictResponse> => {
    const response = await apiClient.post<LiverPredictResponse>(
      '/api/assessments/liver/predict',
      request,
      { timeout: 35000 } // Allow extra time for first ML model load
    );
    return response.data;
  },
};

export default liverPredictionService;
