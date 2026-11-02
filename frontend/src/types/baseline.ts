export interface PersonalBaseline {
  id: number;
  patientId: number;
  metricType: string;
  baselineValue: number;
  variability: number;
  observationCount: number;
  calculatedAt: string;
  updatedAt?: string;
}
