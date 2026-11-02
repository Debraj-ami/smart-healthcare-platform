import apiClient from './api';
import { TrendAnalysis } from '../types/trend';

export const trendService = {
  getTrendsByPatient: async (patientId: number): Promise<TrendAnalysis[]> => {
    const response = await apiClient.get<TrendAnalysis[]>(`/api/trends/patient/${patientId}`);
    return response.data;
  },

  getTrendsByMetric: async (patientId: number, metricType: string): Promise<TrendAnalysis[]> => {
    const response = await apiClient.get<TrendAnalysis[]>(
      `/api/trends/patient/${patientId}/${metricType}`
    );
    return response.data;
  },

  analyzeTrend: async (patientId: number, metricType: string): Promise<TrendAnalysis> => {
    const response = await apiClient.post<TrendAnalysis>(
      `/api/trends/patient/${patientId}/${metricType}/analyze`
    );
    return response.data;
  },
};

export default trendService;
