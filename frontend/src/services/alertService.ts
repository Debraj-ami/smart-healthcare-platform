import apiClient from './api';
import { Alert } from '../types/alert';

export const alertService = {
  getAlertsByPatient: async (patientId: number): Promise<Alert[]> => {
    const response = await apiClient.get<Alert[]>(`/api/alerts/patient/${patientId}`);
    return response.data;
  },
};

export default alertService;
