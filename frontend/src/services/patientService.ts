import apiClient from './api';
import { PatientProfile, PatientProfileRequest } from '../types/patient';

export const patientService = {
  getPatientProfile: async (id: number): Promise<PatientProfile> => {
    const response = await apiClient.get<PatientProfile>(`/api/patients/${id}`);
    return response.data;
  },

  savePatientProfile: async (id: number, data: PatientProfileRequest): Promise<PatientProfile> => {
    const response = await apiClient.post<PatientProfile>(`/api/patients/${id}/profile`, data);
    return response.data;
  },
};

export default patientService;
