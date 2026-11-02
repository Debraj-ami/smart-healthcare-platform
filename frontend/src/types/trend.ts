export interface HistoricalContext {
  observationCount: number;
  historicalMinimum: number;
  historicalMaximum: number;
  historicalMean: number;
  recentMean: number;
  differenceFromHistoricalMean: number;
  differenceFromRecentMean: number;
}

export interface TemporalContext {
  observationDate: string;
  observationTime: string;
  hourOfDay: number;
  dayOfWeek: string;
  timeOfDay: string;
}

export interface ActivityContext {
  activityLevel: string;
  activityMinutes?: number | null;
}

export type TrendDirection = 'INCREASING' | 'DECREASING' | 'STABLE';

export interface TrendAnalysis {
  id: number;
  patientId: number;
  metricType: string;
  baselineId?: number;
  currentValue: number;
  baselineValue?: number;
  variability?: number;
  deviationScore?: number;
  absoluteDeviation?: number;
  deviationFromBaseline?: number;
  previousValue?: number | null;
  valueChangeFromPrevious?: number | null;
  percentageChangeFromPrevious?: number | null;
  trendDirection: TrendDirection;
  persistence?: number;
  historicalContext?: HistoricalContext | null;
  temporalContext?: TemporalContext | null;
  activityContext?: ActivityContext | null;
  analyzedAt: string;
}
