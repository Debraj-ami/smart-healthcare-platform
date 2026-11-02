import React, { useState, useEffect } from 'react';
import { usePatient } from '../../context/PatientContext';
import {
  Activity,
  FileCheck2,
  Bell,
  TrendingUp,
  ArrowUpRight,
  Plus,
  Sparkles,
} from 'lucide-react';
import { HealthObservation } from '../../types/observation';
import { DiseaseAssessment } from '../../types/assessment';
import { Alert } from '../../types/alert';
import { TrendAnalysis } from '../../types/trend';
import observationService from '../../services/observationService';
import assessmentService from '../../services/assessmentService';
import alertService from '../../services/alertService';
import trendService from '../../services/trendService';
import { formatMetricName, formatDate, getDefaultUnit } from '../../utils/formatters';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import TrendBadge from '../../components/monitoring/TrendBadge';
import ObservationTrendChart from '../../components/charts/ObservationTrendChart';
import AddObservationModal from '../../components/health/AddObservationModal';
import { CardSkeleton } from '../../components/common/LoadingSkeleton';
import { Link } from 'react-router-dom';

export const DashboardPage: React.FC = () => {
  const { activePatientId, currentUser } = usePatient();

  const [observations, setObservations] = useState<HealthObservation[]>([]);
  const [assessments, setAssessments] = useState<DiseaseAssessment[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [latestTrend, setLatestTrend] = useState<TrendAnalysis | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);

  const loadDashboardData = async () => {
    setIsLoading(true);
    try {
      // 1. Fetch real observations
      const obsData = await observationService.getObservationsByPatient(activePatientId);
      setObservations(obsData);

      // 2. Fetch real assessments
      try {
        const assData = await assessmentService.getAssessmentsByPatient(activePatientId);
        setAssessments(assData);
      } catch {
        setAssessments([]);
      }

      // 3. Fetch real alerts
      try {
        const alertData = await alertService.getAlertsByPatient(activePatientId);
        setAlerts(alertData);
      } catch {
        setAlerts([]);
      }

      // 4. Fetch latest trend analysis
      try {
        const trends = await trendService.getTrendsByPatient(activePatientId);
        if (trends && trends.length > 0) {
          setLatestTrend(trends[0]); // sorted DESC by analyzedAt
        } else {
          setLatestTrend(null);
        }
      } catch {
        setLatestTrend(null);
      }
    } catch (err) {
      console.error('Failed to load dashboard metrics', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, [activePatientId]);

  const patientName = currentUser
    ? `${currentUser.firstName}`
    : `Patient #${activePatientId}`;

  // Metric flexibility: discover real metric types from patient observations
  const availableMetrics = Array.from(new Set(observations.map((o) => o.metricType)));
  const [selectedDashboardMetric, setSelectedDashboardMetric] = useState<string>('');

  const activeMetric = selectedDashboardMetric && availableMetrics.includes(selectedDashboardMetric)
    ? selectedDashboardMetric
    : latestTrend?.metricType && availableMetrics.includes(latestTrend.metricType)
    ? latestTrend.metricType
    : availableMetrics[0] || 'GLUCOSE';

  const chartObservations = observations.filter((o) => o.metricType === activeMetric);
  const primaryUnit = chartObservations[0]?.unit || getDefaultUnit(activeMetric);

  return (
    <div className="space-y-6">
      {/* 1. Welcome Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white rounded-2xl p-6 border border-slate-200 shadow-card">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2">
            <span>Welcome back, {patientName}</span>
            <span className="text-2xl">👋</span>
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Here is your authentic health overview and personalized monitoring report.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="text-right hidden sm:block">
            <span className="text-xs font-semibold text-slate-700 block">
              {new Date().toLocaleDateString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'short',
                day: 'numeric',
              })}
            </span>
            <span className="text-[11px] text-teal-600 font-medium">Patient #{activePatientId}</span>
          </div>

          <Button
            variant="teal"
            size="sm"
            onClick={() => setIsModalOpen(true)}
            leftIcon={<Plus className="w-4 h-4" />}
          >
            Add Observation
          </Button>
        </div>
      </div>

      {/* 2. KPI Summary Cards (Real Backend Metrics) */}
      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <CardSkeleton count={4} />
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Total Observations Card */}
          <Card className="p-5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Total Observations
              </span>
              <div className="w-9 h-9 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center">
                <Activity className="w-5 h-5" />
              </div>
            </div>
            <div className="mt-3 flex items-baseline gap-2">
              <span className="text-2xl font-bold text-slate-900">{observations.length}</span>
              <span className="text-xs text-slate-400">records</span>
            </div>
            <Link
              to="/health"
              className="text-xs font-semibold text-blue-600 hover:underline flex items-center gap-1 mt-3"
            >
              <span>View observations</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </Link>
          </Card>

          {/* Stored Assessments Card */}
          <Card className="p-5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                ML Risk Assessments
              </span>
              <div className="w-9 h-9 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center">
                <FileCheck2 className="w-5 h-5" />
              </div>
            </div>
            <div className="mt-3 flex items-baseline gap-2">
              <span className="text-2xl font-bold text-slate-900">{assessments.length}</span>
              <span className="text-xs text-slate-400">reports</span>
            </div>
            <Link
              to="/assessments"
              className="text-xs font-semibold text-teal-600 hover:underline flex items-center gap-1 mt-3"
            >
              <span>View reports</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </Link>
          </Card>

          {/* Active Alerts Card */}
          <Card className="p-5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Active Alerts
              </span>
              <div
                className={`w-9 h-9 rounded-xl flex items-center justify-center ${
                  alerts.length > 0 ? 'bg-rose-50 text-rose-600' : 'bg-slate-50 text-slate-400'
                }`}
              >
                <Bell className="w-5 h-5" />
              </div>
            </div>
            <div className="mt-3 flex items-baseline gap-2">
              <span className="text-2xl font-bold text-slate-900">{alerts.length}</span>
              <span className="text-xs text-slate-400">notifications</span>
            </div>
            <Link
              to="/alerts"
              className="text-xs font-semibold text-rose-600 hover:underline flex items-center gap-1 mt-3"
            >
              <span>{alerts.length > 0 ? 'Review alerts' : 'View alert log'}</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </Link>
          </Card>

          {/* Monitoring / Trend Status Card */}
          <Card className="p-5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Monitoring Status
              </span>
              <div className="w-9 h-9 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center">
                <TrendingUp className="w-5 h-5" />
              </div>
            </div>
            <div className="mt-2.5 space-y-1">
              {latestTrend ? (
                <>
                  <div className="flex items-center gap-2">
                    <TrendBadge direction={latestTrend.trendDirection} showPrefix={true} />
                  </div>
                  {latestTrend.persistence != null && (
                    <span className="text-xs text-slate-500 font-medium block">
                      {latestTrend.persistence} consecutive observations
                    </span>
                  )}
                </>
              ) : (
                <span className="text-sm font-bold text-slate-700">No active trend</span>
              )}
            </div>
            <Link
              to="/monitoring"
              className="text-xs font-semibold text-emerald-600 hover:underline flex items-center gap-1 mt-3"
            >
              <span>Detailed trends</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </Link>
          </Card>
        </div>
      )}

      {/* 3. Main Dashboard Body: Chart + AI Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Real Observation Chart (8 cols) */}
        <div className="lg:col-span-8 bg-white rounded-2xl p-6 border border-slate-200 shadow-card">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
            <div>
              <h3 className="text-base font-bold text-slate-800">
                {formatMetricName(activeMetric)} Trend
              </h3>
              <p className="text-xs text-slate-400">
                Authentic chronological measurements from verified backend observations
              </p>
            </div>
            <div className="flex items-center gap-2">
              {availableMetrics.length > 1 && (
                <select
                  value={activeMetric}
                  onChange={(e) => setSelectedDashboardMetric(e.target.value)}
                  className="text-xs font-semibold py-1.5 px-2.5 bg-slate-50 border border-slate-200 rounded-lg text-slate-700 focus:outline-none focus:border-teal-500 cursor-pointer"
                >
                  {availableMetrics.map((m) => (
                    <option key={m} value={m}>
                      {formatMetricName(m)}
                    </option>
                  ))}
                </select>
              )}
              <Link to="/monitoring">
                <Button variant="ghost" size="sm">
                  Full Analytics
                </Button>
              </Link>
            </div>
          </div>

          <ObservationTrendChart
            observations={chartObservations}
            baselineValue={latestTrend?.metricType === activeMetric ? latestTrend.baselineValue : null}
            metricName={formatMetricName(activeMetric)}
            unit={primaryUnit}
          />
        </div>

        {/* Real Contextual Insights (4 cols) */}
        <div className="lg:col-span-4 bg-white rounded-2xl p-6 border border-slate-200 shadow-card space-y-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-teal-50 text-teal-600 flex items-center justify-center">
              <Sparkles className="w-4 h-4" />
            </div>
            <h3 className="text-base font-bold text-slate-800">Personal Insights</h3>
          </div>

          {latestTrend ? (
            <div className="space-y-3 text-xs leading-relaxed">
              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
                <span className="font-semibold text-slate-800 block mb-1">
                  Baseline Comparison ({formatMetricName(latestTrend.metricType)})
                </span>
                <p className="text-slate-600">
                  Current measurement is{' '}
                  <strong className="text-slate-900">
                    {latestTrend.currentValue} {getDefaultUnit(latestTrend.metricType)}
                  </strong>
                  {latestTrend.baselineValue != null && (
                    <span>
                      {' '}versus your personal baseline of{' '}
                      <strong className="text-teal-700">
                        {latestTrend.baselineValue.toFixed(1)} {getDefaultUnit(latestTrend.metricType)}
                      </strong>.
                    </span>
                  )}
                </p>
                {latestTrend.deviationFromBaseline != null && (
                  <p className="mt-1 text-slate-500">
                    Net deviation: {latestTrend.deviationFromBaseline > 0 ? '+' : ''}
                    {latestTrend.deviationFromBaseline.toFixed(1)} {getDefaultUnit(latestTrend.metricType)} (absolute: {latestTrend.absoluteDeviation?.toFixed(1)} {getDefaultUnit(latestTrend.metricType)}).
                  </p>
                )}
              </div>

              {latestTrend.historicalContext && (
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
                  <span className="font-semibold text-slate-800 block mb-1">
                    Historical Averages ({formatMetricName(latestTrend.metricType)})
                  </span>
                  <p className="text-slate-600">
                    Overall mean: <strong>{latestTrend.historicalContext.historicalMean.toFixed(1)} {getDefaultUnit(latestTrend.metricType)}</strong> across {latestTrend.historicalContext.observationCount} observations. Recent window mean: <strong>{latestTrend.historicalContext.recentMean.toFixed(1)} {getDefaultUnit(latestTrend.metricType)}</strong>.
                  </p>
                </div>
              )}

              {latestTrend.activityContext?.activityLevel && (
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
                  <span className="font-semibold text-slate-800 block mb-1">
                    Recorded Activity Context
                  </span>
                  <p className="text-slate-600">
                    Observation logged during <strong>{latestTrend.activityContext.activityLevel.toLowerCase()}</strong> state
                    {latestTrend.activityContext.activityMinutes ? ` (${latestTrend.activityContext.activityMinutes} minutes duration)` : ''}.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="py-8 text-center bg-slate-50/60 rounded-xl border border-dashed border-slate-200 p-4">
              <Activity className="w-8 h-8 text-slate-300 mx-auto mb-2" />
              <p className="text-xs font-medium text-slate-600">No Trend Insights Yet</p>
              <p className="text-[11px] text-slate-400 mt-1">
                Record at least two observations and analyze your trend to generate personal statistical insights.
              </p>
              <Button
                variant="teal"
                size="sm"
                className="mt-3"
                onClick={() => setIsModalOpen(true)}
              >
                Add Observation
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* 4. Recent Activity Log */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-bold text-slate-800">Recent Biometric Observations</h3>
          <Link to="/health">
            <Button variant="ghost" size="sm">
              View All ({observations.length})
            </Button>
          </Link>
        </div>

        {observations.length === 0 ? (
          <p className="text-xs text-slate-400 py-6 text-center">
            No health observations recorded yet. Click 'Add Observation' above to log your first vital.
          </p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            {observations.slice(0, 6).map((obs) => (
              <div
                key={obs.id}
                className="p-3.5 rounded-xl bg-slate-50/70 border border-slate-100 flex items-center justify-between"
              >
                <div>
                  <span className="text-xs font-bold text-slate-800 block">
                    {formatMetricName(obs.metricType)}
                  </span>
                  <span className="text-[11px] text-slate-400">{formatDate(obs.loggedAt)}</span>
                </div>
                <div className="text-right">
                  <span className="text-sm font-bold text-slate-900">{obs.metricValue}</span>
                  <span className="text-[11px] text-slate-500 ml-1 font-medium">{obs.unit}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Modal for adding observation */}
      <AddObservationModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        patientId={activePatientId}
        onObservationAdded={loadDashboardData}
      />
    </div>
  );
};

export default DashboardPage;
