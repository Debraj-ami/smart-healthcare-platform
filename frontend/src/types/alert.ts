export type AlertSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type AlertStatus = 'ACTIVE' | 'ACKNOWLEDGED' | 'RESOLVED';

export interface Alert {
  id: number;
  patientId: number;
  assessmentId?: number | null;
  trendAnalysisId?: number | null;
  alertType: string;
  severity: AlertSeverity;
  message: string;
  status: AlertStatus;
  createdAt: string;
}
