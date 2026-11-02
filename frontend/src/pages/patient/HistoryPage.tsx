import React, { useState, useEffect } from 'react';
import { usePatient } from '../../context/PatientContext';
import {
  History,
  Activity,
  FileCheck2,
  TrendingUp,
  Bell,
  Search,
  RefreshCw,
  Clock,
  ArrowUpDown,
} from 'lucide-react';
import { HealthObservation } from '../../types/observation';
import { DiseaseAssessment } from '../../types/assessment';
import { TrendAnalysis } from '../../types/trend';
import { Alert } from '../../types/alert';
import observationService from '../../services/observationService';
import assessmentService from '../../services/assessmentService';
import trendService from '../../services/trendService';
import alertService from '../../services/alertService';
import { formatDateTime, formatMetricName, getAlertSeverityBadge } from '../../utils/formatters';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import Badge from '../../components/common/Badge';
import TrendBadge from '../../components/monitoring/TrendBadge';

export const HistoryPage: React.FC = () => {
  const { activePatientId } = usePatient();
  const [activeTab, setActiveTab] = useState<'observations' | 'assessments' | 'trends' | 'alerts'>(
    'observations'
  );

  const [observations, setObservations] = useState<HealthObservation[]>([]);
  const [assessments, setAssessments] = useState<DiseaseAssessment[]>([]);
  const [trends, setTrends] = useState<TrendAnalysis[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [searchTerm, setSearchTerm] = useState<string>('');

  const loadAllHistory = async () => {
    setIsLoading(true);
    try {
      const [obs, ass, tr, al] = await Promise.allSettled([
        observationService.getObservationsByPatient(activePatientId),
        assessmentService.getAssessmentsByPatient(activePatientId),
        trendService.getTrendsByPatient(activePatientId),
        alertService.getAlertsByPatient(activePatientId),
      ]);

      if (obs.status === 'fulfilled') setObservations(obs.value);
      if (ass.status === 'fulfilled') setAssessments(ass.value);
      if (tr.status === 'fulfilled') setTrends(tr.value);
      if (al.status === 'fulfilled') setAlerts(al.value);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadAllHistory();
  }, [activePatientId]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
            Patient Medical History
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Audit logs and longitudinal health records for Patient #{activePatientId}.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={loadAllHistory}
          leftIcon={<RefreshCw className="w-4 h-4" />}
        >
          Refresh All
        </Button>
      </div>

      {/* Tabs matching design reference #12 */}
      <div className="flex items-center justify-between border-b border-slate-200">
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('observations')}
            className={`flex items-center gap-2 px-4 py-3 text-xs sm:text-sm font-semibold border-b-2 transition-all ${
              activeTab === 'observations'
                ? 'border-teal-600 text-teal-600'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            <Activity className="w-4 h-4" />
            <span>Observations ({observations.length})</span>
          </button>

          <button
            onClick={() => setActiveTab('assessments')}
            className={`flex items-center gap-2 px-4 py-3 text-xs sm:text-sm font-semibold border-b-2 transition-all ${
              activeTab === 'assessments'
                ? 'border-teal-600 text-teal-600'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            <FileCheck2 className="w-4 h-4" />
            <span>Assessments ({assessments.length})</span>
          </button>

          <button
            onClick={() => setActiveTab('trends')}
            className={`flex items-center gap-2 px-4 py-3 text-xs sm:text-sm font-semibold border-b-2 transition-all ${
              activeTab === 'trends'
                ? 'border-teal-600 text-teal-600'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            <TrendingUp className="w-4 h-4" />
            <span>Trends ({trends.length})</span>
          </button>

          <button
            onClick={() => setActiveTab('alerts')}
            className={`flex items-center gap-2 px-4 py-3 text-xs sm:text-sm font-semibold border-b-2 transition-all ${
              activeTab === 'alerts'
                ? 'border-teal-600 text-teal-600'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            <Bell className="w-4 h-4" />
            <span>Alerts ({alerts.length})</span>
          </button>
        </div>
      </div>

      {/* Tab 1: Observations Table */}
      {activeTab === 'observations' && (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50/80 text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-3.5 px-4">Date & Time</th>
                  <th className="py-3.5 px-4">Metric</th>
                  <th className="py-3.5 px-4 text-right">Value</th>
                  <th className="py-3.5 px-4">Unit</th>
                  <th className="py-3.5 px-4">Activity Context</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {observations.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-slate-400">
                      No observations recorded yet.
                    </td>
                  </tr>
                ) : (
                  observations.map((obs) => (
                    <tr key={obs.id} className="hover:bg-slate-50/60 transition-colors">
                      <td className="py-3.5 px-4 text-slate-600 font-medium">
                        {formatDateTime(obs.loggedAt)}
                      </td>
                      <td className="py-3.5 px-4 font-bold text-slate-800">
                        {formatMetricName(obs.metricType)}
                      </td>
                      <td className="py-3.5 px-4 text-right font-bold text-slate-900">
                        {obs.metricValue}
                      </td>
                      <td className="py-3.5 px-4 text-slate-500 font-medium">{obs.unit}</td>
                      <td className="py-3.5 px-4">
                        {obs.activityLevel && obs.activityLevel !== 'UNKNOWN' ? (
                          <Badge variant="teal" size="sm">
                            {obs.activityLevel.toLowerCase()}{' '}
                            {obs.activityMinutes ? `(${obs.activityMinutes}m)` : ''}
                          </Badge>
                        ) : (
                          <span className="text-slate-400">—</span>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Tab 2: Assessments Table */}
      {activeTab === 'assessments' && (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50/80 text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-3.5 px-4">Assessed Date</th>
                  <th className="py-3.5 px-4">Disease Screening</th>
                  <th className="py-3.5 px-4">Risk Level</th>
                  <th className="py-3.5 px-4 text-right">Risk Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {assessments.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="py-8 text-center text-slate-400">
                      No ML screening assessments stored in database.
                    </td>
                  </tr>
                ) : (
                  assessments.map((ass) => (
                    <tr key={ass.id} className="hover:bg-slate-50/60 transition-colors">
                      <td className="py-3.5 px-4 text-slate-600 font-medium">
                        {formatDateTime(ass.assessedAt)}
                      </td>
                      <td className="py-3.5 px-4 font-bold text-slate-800">{ass.diseaseType}</td>
                      <td className="py-3.5 px-4">
                        {ass.riskLevel ? (
                          <Badge
                            variant={
                              ass.riskLevel === 'LOW'
                                ? 'success'
                                : ass.riskLevel === 'HIGH' || ass.riskLevel === 'CRITICAL'
                                ? 'danger'
                                : 'warning'
                            }
                          >
                            {ass.riskLevel}
                          </Badge>
                        ) : (
                          <span className="text-slate-400 font-mono text-[11px]">—</span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right font-bold text-slate-900">
                        {ass.riskScore.toFixed(3)}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Tab 3: Trends Table */}
      {activeTab === 'trends' && (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50/80 text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-3.5 px-4">Snapshot Time</th>
                  <th className="py-3.5 px-4">Metric</th>
                  <th className="py-3.5 px-4 text-right">Current</th>
                  <th className="py-3.5 px-4 text-right">Baseline</th>
                  <th className="py-3.5 px-4 text-right">Signed / Abs Dev</th>
                  <th className="py-3.5 px-4">Direction</th>
                  <th className="py-3.5 px-4">Persistence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {trends.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-8 text-center text-slate-400">
                      No trend snapshots stored in database.
                    </td>
                  </tr>
                ) : (
                  trends.map((t) => (
                    <tr key={t.id} className="hover:bg-slate-50/60 transition-colors">
                      <td className="py-3.5 px-4 text-slate-600 font-medium">
                        {formatDateTime(t.analyzedAt)}
                      </td>
                      <td className="py-3.5 px-4 font-bold text-slate-800">
                        {formatMetricName(t.metricType)}
                      </td>
                      <td className="py-3.5 px-4 text-right font-bold text-slate-900">
                        {t.currentValue}
                      </td>
                      <td className="py-3.5 px-4 text-right text-teal-700 font-semibold">
                        {t.baselineValue != null ? t.baselineValue.toFixed(1) : '—'}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono text-[11px]">
                        {t.deviationFromBaseline != null
                          ? `${t.deviationFromBaseline > 0 ? '+' : ''}${t.deviationFromBaseline.toFixed(1)} / ${t.absoluteDeviation?.toFixed(1)}`
                          : '—'}
                      </td>
                      <td className="py-3.5 px-4">
                        <TrendBadge direction={t.trendDirection} />
                      </td>
                      <td className="py-3.5 px-4 font-semibold text-slate-700">
                        {t.persistence ?? 1}x
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Tab 4: Alerts Table */}
      {activeTab === 'alerts' && (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50/80 text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-3.5 px-4">Created Time</th>
                  <th className="py-3.5 px-4">Alert Type</th>
                  <th className="py-3.5 px-4">Severity</th>
                  <th className="py-3.5 px-4">Message</th>
                  <th className="py-3.5 px-4">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {alerts.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-slate-400">
                      No alert records found.
                    </td>
                  </tr>
                ) : (
                  alerts.map((al) => {
                    const badge = getAlertSeverityBadge(al.severity);
                    return (
                      <tr key={al.id} className="hover:bg-slate-50/60 transition-colors">
                        <td className="py-3.5 px-4 text-slate-600 font-medium">
                          {formatDateTime(al.createdAt)}
                        </td>
                        <td className="py-3.5 px-4 font-bold text-slate-800">{al.alertType}</td>
                        <td className="py-3.5 px-4">
                          <span
                            className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border ${badge.bg} ${badge.text} ${badge.border}`}
                          >
                            {al.severity}
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-slate-700 max-w-sm leading-relaxed">
                          {al.message}
                        </td>
                        <td className="py-3.5 px-4 font-medium text-slate-500">{al.status}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
};

export default HistoryPage;
