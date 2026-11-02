import React, { useState, useEffect } from 'react';
import { usePatient } from '../../context/PatientContext';
import { Bell, AlertTriangle, CheckCircle2, ShieldAlert, Clock, RefreshCw } from 'lucide-react';
import { Alert, AlertSeverity } from '../../types/alert';
import alertService from '../../services/alertService';
import { formatDateTime, getAlertSeverityBadge } from '../../utils/formatters';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import Badge from '../../components/common/Badge';

export const AlertsPage: React.FC = () => {
  const { activePatientId } = usePatient();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [filterSeverity, setFilterSeverity] = useState<string>('ALL');
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fetchAlerts = async () => {
    setIsLoading(true);
    try {
      const data = await alertService.getAlertsByPatient(activePatientId);
      setAlerts(data);
    } catch (err) {
      console.error('Failed to load alerts', err);
      setAlerts([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, [activePatientId]);

  const filteredAlerts = alerts.filter((a) => {
    if (filterSeverity === 'ALL') return true;
    return a.severity === filterSeverity;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
            Alerts & Notifications
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Stay informed about biometric threshold deviations and clinical notifications for Patient #{activePatientId}.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={fetchAlerts}
          leftIcon={<RefreshCw className="w-4 h-4" />}
        >
          Refresh Alerts
        </Button>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-200 pb-3">
        <button
          onClick={() => setFilterSeverity('ALL')}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
            filterSeverity === 'ALL'
              ? 'bg-slate-900 text-white'
              : 'text-slate-600 hover:bg-slate-100'
          }`}
        >
          All ({alerts.length})
        </button>
        <button
          onClick={() => setFilterSeverity('CRITICAL')}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
            filterSeverity === 'CRITICAL'
              ? 'bg-rose-600 text-white'
              : 'text-slate-600 hover:bg-slate-100'
          }`}
        >
          Critical ({alerts.filter((a) => a.severity === 'CRITICAL').length})
        </button>
        <button
          onClick={() => setFilterSeverity('HIGH')}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
            filterSeverity === 'HIGH'
              ? 'bg-orange-600 text-white'
              : 'text-slate-600 hover:bg-slate-100'
          }`}
        >
          High ({alerts.filter((a) => a.severity === 'HIGH').length})
        </button>
      </div>

      {/* Alerts List / Empty State */}
      {filteredAlerts.length === 0 ? (
        <Card className="p-12 text-center bg-white">
          <div className="w-12 h-12 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto mb-3">
            <CheckCircle2 className="w-6 h-6" />
          </div>
          <h3 className="text-base font-bold text-slate-800">No Active Health Alerts</h3>
          <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1 leading-relaxed">
            There are no active critical or warning alerts recorded for Patient #{activePatientId}. The system will notify you if sustained negative trends or acute threshold deviations are flagged.
          </p>
        </Card>
      ) : (
        <div className="space-y-3">
          {filteredAlerts.map((alert) => {
            const badge = getAlertSeverityBadge(alert.severity);
            return (
              <Card key={alert.id} className="p-5 flex items-start justify-between gap-4">
                <div className="flex items-start gap-3.5">
                  <div
                    className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5 ${
                      alert.severity === 'CRITICAL'
                        ? 'bg-rose-100 text-rose-700'
                        : alert.severity === 'HIGH'
                        ? 'bg-orange-100 text-orange-700'
                        : 'bg-amber-100 text-amber-700'
                    }`}
                  >
                    <AlertTriangle className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-bold text-slate-900">{alert.alertType}</span>
                      <span
                        className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border ${badge.bg} ${badge.text} ${badge.border}`}
                      >
                        {alert.severity}
                      </span>
                      <span className="text-[11px] text-slate-400 font-medium">
                        Status: {alert.status}
                      </span>
                    </div>
                    <p className="text-xs text-slate-600 leading-relaxed">{alert.message}</p>
                    <div className="flex items-center gap-1.5 text-[11px] text-slate-400 mt-2">
                      <Clock className="w-3.5 h-3.5" />
                      <span>{formatDateTime(alert.createdAt)}</span>
                    </div>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default AlertsPage;
