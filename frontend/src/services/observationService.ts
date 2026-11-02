import apiClient from './api';
import { HealthObservation, HealthObservationRequest } from '../types/observation';

export const observationService = {
  recordObservation: async (data: HealthObservationRequest): Promise<HealthObservation> => {
    const response = await apiClient.post<HealthObservation>('/api/observations', data);
    return response.data;
  },

  getObservationsByPatient: async (patientId: number): Promise<HealthObservation[]> => {
    const response = await apiClient.get<HealthObservation[]>(`/api/observations/patient/${patientId}`);
    return response.data;
  },
};

export default observationService;
