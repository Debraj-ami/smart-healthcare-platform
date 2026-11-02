import apiClient from './api';
import { PersonalBaseline } from '../types/baseline';

export const baselineService = {
  getBaselinesByPatient: async (patientId: number): Promise<PersonalBaseline[]> => {
    const response = await apiClient.get<PersonalBaseline[]>(`/api/baselines/patient/${patientId}`);
    return response.data;
  },

  getBaselineByMetric: async (patientId: number, metricType: string): Promise<PersonalBaseline> => {
    const response = await apiClient.get<PersonalBaseline>(`/api/baselines/patient/${patientId}/${metricType}`);
    return response.data;
  },

  calculateBaseline: async (patientId: number, metricType: string): Promise<PersonalBaseline> => {
    const response = await apiClient.post<PersonalBaseline>(
      `/api/baselines/patient/${patientId}/${metricType}/calculate`
    );
    return response.data;
  },
};

export default baselineService;
