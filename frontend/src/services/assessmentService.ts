import apiClient from './api';
import { DiseaseAssessment, DiseaseAssessmentRequest } from '../types/assessment';

export const assessmentService = {
  getAssessmentsByPatient: async (patientId: number): Promise<DiseaseAssessment[]> => {
    const response = await apiClient.get<DiseaseAssessment[]>(`/api/assessments/patient/${patientId}`);
    return response.data;
  },

  saveAssessment: async (data: DiseaseAssessmentRequest): Promise<DiseaseAssessment> => {
    const response = await apiClient.post<DiseaseAssessment>('/api/assessments', data);
    return response.data;
  },
};

export default assessmentService;
