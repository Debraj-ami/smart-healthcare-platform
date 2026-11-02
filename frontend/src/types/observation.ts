export type MetricType =
  | 'GLUCOSE'
  | 'HEART_RATE'
  | 'BLOOD_PRESSURE_SYSTOLIC'
  | 'BLOOD_PRESSURE_DIASTOLIC'
  | 'TEMPERATURE'
  | 'OXYGEN_SATURATION'
  | 'WEIGHT'
  | 'HEIGHT';

export type ActivityLevel = 'UNKNOWN' | 'RESTING' | 'MODERATE' | 'INTENSE';

export interface HealthObservation {
  id: number;
  patientId: number;
  metricType: MetricType | string;
  metricValue: number;
  unit: string;
  loggedAt: string;
  activityLevel?: ActivityLevel;
  activityMinutes?: number | null;
}

export interface HealthObservationRequest {
  patientId: number;
  metricType: MetricType | string;
  metricValue: number;
  unit: string;
  activityLevel?: ActivityLevel;
  activityMinutes?: number | null;
}
