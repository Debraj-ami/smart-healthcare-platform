export type DiseaseType = 'DIABETES' | 'HEART_DISEASE' | 'PARKINSONS' | 'LIVER_DISEASE' | string;
export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | string;

export interface DiseaseAssessment {
  id: number;
  patientId: number;
  diseaseType: DiseaseType;
  riskScore: number;
  riskLevel: RiskLevel | null;
  inputFeatures?: string | null;
  assessedAt: string;
}

export interface DiseaseAssessmentRequest {
  patientId: number;
  diseaseType: DiseaseType;
  riskScore: number;
  riskLevel: RiskLevel;
  inputFeatures?: string;
}

export interface LiverDiseaseFeatures {
  age: string;
  gender: 'Male' | 'Female' | '';
  totalBilirubin: string;
  directBilirubin: string;
  alkalinePhosphatase: string;
  sgpt: string;
  sgot: string;
  totalProteins: string;
  albumin: string;
  agRatio: string;
}

export interface HeartDiseaseFeatures {
  age: string;
  sex: string;
  cp: string;
  trestbps: string;
  chol: string;
  fbs: string;
  restecg: string;
  thalach: string;
  exang: string;
  oldpeak: string;
  slope: string;
  ca: string;
  thal: string;
}

