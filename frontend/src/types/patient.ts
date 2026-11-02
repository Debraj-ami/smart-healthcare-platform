export interface PatientProfile {
  id?: number;
  patientId: number;
  bloodGroup?: string;
  heightCm?: number;
  weightKg?: number;
  lifestyleContext?: string;
  medicalHistory?: string;
  createdAt?: string;
  updatedAt?: string;
}

export interface PatientProfileRequest {
  bloodGroup?: string;
  heightCm?: number;
  weightKg?: number;
  lifestyleContext?: string;
  medicalHistory?: string;
}
