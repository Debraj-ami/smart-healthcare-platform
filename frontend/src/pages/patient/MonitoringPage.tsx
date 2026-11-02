import React, { useState, useEffect } from 'react';
import { usePatient } from '../../context/PatientContext';
import {
  TrendingUp,
  RefreshCw,
  Calculator,
  Activity,
  AlertCircle,
  CheckCircle2,
  Calendar,
} from 'lucide-react';
import { MetricType, HealthObservation } from '../../types/observation';
import { TrendAnalysis } from '../../types/trend';
import { PersonalBaseline } from '../../types/baseline';
import observationService from '../../services/observationService';
import baselineService from '../../services/baselineService';
import trendService from '../../services/trendService';
import { formatMetricName, getDefaultUnit } from '../../utils/formatters';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import ObservationTrendChart from '../../components/charts/ObservationTrendChart';
import ContextStatsGrid from '../../components/monitoring/ContextStatsGrid';
import { useChatAssistant } from '../../context/ChatAssistantContext';

const AVAILABLE_METRICS: MetricType[] = [
  'GLUCOSE',
  'HEART_RATE',
  'BLOOD_PRESSURE_SYSTOLIC',
  'BLOOD_PRESSURE_DIASTOLIC',
  'TEMPERATURE',
  'OXYGEN_SATURATION',
  'WEIGHT',
];

export const MonitoringPage: React.FC = () => {
  const { activePatientId } = usePatient();
  const { setPageContext } = useChatAssistant();
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('GLUCOSE');

  // Register selected monitoring metric context with Smart Health AI assistant
  useEffect(() => {
    setPageContext({
      source: 'monitoring',
      metric: selectedMetric,
    });
    return () => {
      setPageContext(null);
    };
  }, [selectedMetric, setPageContext]);

  const [observations, setObservations] = useState<HealthObservation[]>([]);
  const [baseline, setBaseline] = useState<PersonalBaseline | null>(null);
  const [trendAnalysis, setTrendAnalysis] = useState<TrendAnalysis | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isActionLoading, setIsActionLoading] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const [patientMetrics, setPatientMetrics] = useState<string[]>([]);

  const loadMetricData = async () => {
    setIsLoading(true);
    try {
      // 1. Fetch observations for patient
      const allObs = await observationService.getObservationsByPatient(activePatientId);
      const uniqueObsMetrics = Array.from(new Set(allObs.map((o) => o.metricType)));
      setPatientMetrics(uniqueObsMetrics);

      const filtered = allObs.filter((o) => o.metricType === selectedMetric);
      setObservations(filtered);

      // 2. Fetch baseline for metric
      try {
        const b = await baselineService.getBaselineByMetric(activePatientId, selectedMetric);
        setBaseline(b);
      } catch {
        setBaseline(null);
      }

      // 3. Fetch trends for metric
      try {
        const trends = await trendService.getTrendsByMetric(activePatientId, selectedMetric);
        if (trends && trends.length > 0) {
          setTrendAnalysis(trends[0]); // most recent
        } else {
          setTrendAnalysis(null);
        }
      } catch {
        setTrendAnalysis(null);
      }
    } catch (err: any) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    setStatusMessage(null);
    loadMetricData();
  }, [activePatientId, selectedMetric]);

  // Recalculate baseline endpoint
  const handleRecalculateBaseline = async () => {
    setIsActionLoading(true);
    setStatusMessage(null);
    try {
      const b = await baselineService.calculateBaseline(activePatientId, selectedMetric);
      setBaseline(b);
      setStatusMessage({
        type: 'success',
        text: `Baseline updated: ${b.baselineValue.toFixed(1)} ${getDefaultUnit(selectedMetric)} (n=${b.observationCount})`,
      });
      await loadMetricData();
    } catch (err: any) {
      setStatusMessage({
        type: 'error',
        text: err.message || 'Failed to recalculate baseline. Check if sufficient observations exist.',
      });
    } finally {
      setIsActionLoading(false);
    }
  };

  // Analyze trend endpoint
  const handleAnalyzeTrend = async () => {
    setIsActionLoading(true);
    setStatusMessage(null);
    try {
      const t = await trendService.analyzeTrend(activePatientId, selectedMetric);
      setTrendAnalysis(t);
      setStatusMessage({
        type: 'success',
        text: `Trend snapshot created: ${t.trendDirection} (Persistence: ${t.persistence}x, Delta: ${t.valueChangeFromPrevious ?? 0})`,
      });
      await loadMetricData();
    } catch (err: any) {
      setStatusMessage({
        type: 'error',
        text: err.message || 'Failed to analyze trend. Ensure at least one observation is recorded.',
      });
    } finally {
      setIsActionLoading(false);
    }
  };

  const unit = observations[0]?.unit || getDefaultUnit(selectedMetric);
  const selectableMetrics = Array.from(new Set([...patientMetrics, ...AVAILABLE_METRICS]));

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white rounded-2xl p-6 border border-slate-200 shadow-card">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
            Health Monitoring & Trends
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Personalized baseline tracking and context-aware trend analytics for Patient #{activePatientId}.
          </p>
        </div>

        {/* Metric Selector Dropdown */}
        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold text-slate-500 hidden sm:inline">
            Metric:
          </label>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value as MetricType)}
            className="text-sm font-semibold py-2 px-3 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 cursor-pointer shadow-sm"
          >
            {selectableMetrics.map((m) => (
              <option key={m} value={m}>
                {formatMetricName(m)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {statusMessage && (
        <div
          className={`p-3.5 rounded-xl text-xs flex items-center gap-2 ${
            statusMessage.type === 'success'
              ? 'bg-emerald-50 border border-emerald-200 text-emerald-800'
              : 'bg-rose-50 border border-rose-200 text-rose-800'
          }`}
        >
          {statusMessage.type === 'success' ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
          ) : (
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
          )}
          <span>{statusMessage.text}</span>
        </div>
      )}

      {/* Action Bar: Calculate Baseline & Analyze Trend */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-4 bg-slate-50 rounded-2xl border border-slate-200">
        <div className="text-xs text-slate-600">
          Showing real data for <strong className="text-slate-900">{formatMetricName(selectedMetric)}</strong> ({observations.length} observations recorded)
        </div>

        <div className="flex items-center gap-2.5">
          <Button
            variant="outline"
            size="sm"
            onClick={handleRecalculateBaseline}
            isLoading={isActionLoading}
            leftIcon={<Calculator className="w-4 h-4" />}
          >
            Recalculate Baseline
          </Button>

          <Button
            variant="teal"
            size="sm"
            onClick={handleAnalyzeTrend}
            isLoading={isActionLoading}
            leftIcon={<TrendingUp className="w-4 h-4" />}
          >
            Analyze New Trend
          </Button>
        </div>
      </div>

      {/* Main Chart Section */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-slate-800">
              {formatMetricName(selectedMetric)} Trend Visualization
            </h3>
            <p className="text-xs text-slate-400">
              Measured observations over time with dashed personal baseline reference
            </p>
          </div>
          {baseline && (
            <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-teal-50 text-teal-700 border border-teal-200">
              Baseline: {baseline.baselineValue.toFixed(1)} {unit} (±{baseline.variability.toFixed(1)})
            </span>
          )}
        </div>

        <ObservationTrendChart
          observations={observations}
          baselineValue={baseline?.baselineValue}
          metricName={formatMetricName(selectedMetric)}
          unit={unit}
        />
      </Card>

      {/* Contextual Stats Grid */}
      {trendAnalysis ? (
        <ContextStatsGrid trend={trendAnalysis} unit={unit} />
      ) : (
        <Card className="p-8 text-center bg-slate-50/50">
          <Activity className="w-10 h-10 text-slate-300 mx-auto mb-2" />
          <h4 className="text-sm font-semibold text-slate-700">No Trend Analysis Stored Yet</h4>
          <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1 mb-4 leading-relaxed">
            Click 'Analyze New Trend' above to evaluate current observation trajectory, baseline deviation, and persistence metrics.
          </p>
          <Button
            variant="teal"
            size="sm"
            onClick={handleAnalyzeTrend}
            isLoading={isActionLoading}
          >
            Analyze Trend Now
          </Button>
        </Card>
      )}
    </div>
  );
};

export default MonitoringPage;
