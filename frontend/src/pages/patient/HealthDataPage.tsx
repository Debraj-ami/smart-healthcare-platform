import React, { useState, useEffect } from 'react';
import { usePatient } from '../../context/PatientContext';
import { Plus, Activity, Heart, Clock, AlertCircle, CheckCircle2, RefreshCw } from 'lucide-react';
import { MetricType, ActivityLevel, HealthObservation } from '../../types/observation';
import observationService from '../../services/observationService';
import { formatMetricName, getDefaultUnit } from '../../utils/formatters';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import ObservationList from '../../components/health/ObservationList';

const SUPPORTED_METRICS: MetricType[] = [
  'GLUCOSE',
  'HEART_RATE',
  'BLOOD_PRESSURE_SYSTOLIC',
  'BLOOD_PRESSURE_DIASTOLIC',
  'TEMPERATURE',
  'OXYGEN_SATURATION',
  'WEIGHT',
  'HEIGHT',
];

export const HealthDataPage: React.FC = () => {
  const { activePatientId } = usePatient();
  const [observations, setObservations] = useState<HealthObservation[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Form state
  const [metricType, setMetricType] = useState<MetricType>('GLUCOSE');
  const [metricValue, setMetricValue] = useState<string>('');
  const [unit, setUnit] = useState<string>('mg/dL');
  const [activityLevel, setActivityLevel] = useState<ActivityLevel>('UNKNOWN');
  const [activityMinutes, setActivityMinutes] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const fetchObservations = async () => {
    setIsLoading(true);
    try {
      const data = await observationService.getObservationsByPatient(activePatientId);
      setObservations(data);
    } catch (err: any) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchObservations();
  }, [activePatientId]);

  const handleMetricChange = (type: MetricType) => {
    setMetricType(type);
    setUnit(getDefaultUnit(type));
  };

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const val = parseFloat(metricValue);
    if (isNaN(val)) {
      setError('Please provide a valid numeric measurement.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setSuccessMessage(null);

    try {
      await observationService.recordObservation({
        patientId: activePatientId,
        metricType,
        metricValue: val,
        unit,
        activityLevel,
        activityMinutes: activityMinutes ? parseInt(activityMinutes, 10) : null,
      });

      setMetricValue('');
      setActivityMinutes('');
      setSuccessMessage(
        `Observation recorded successfully: ${formatMetricName(metricType)} = ${val} ${unit}`
      );
      await fetchObservations();
    } catch (err: any) {
      setError(err.message || 'Failed to submit observation to backend.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
            Health Observations
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Record biometric vital signs and review historical observations for Patient #{activePatientId}.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={fetchObservations}
          leftIcon={<RefreshCw className="w-4 h-4" />}
        >
          Refresh Data
        </Button>
      </div>

      {successMessage && (
        <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
          <span>{successMessage}</span>
        </div>
      )}

      {error && (
        <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Grid: Left Form (5 cols) + Right Observations (7 cols) - matching UI reference #7 */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Add Health Observation Form */}
        <div className="lg:col-span-5 bg-white rounded-2xl p-6 border border-slate-200 shadow-card">
          <div className="flex items-center gap-2.5 pb-4 border-b border-slate-100 mb-5">
            <div className="w-9 h-9 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center font-bold">
              <Plus className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">Add Health Observation</h2>
              <p className="text-xs text-slate-400">Directly persisted to Spring Boot backend</p>
            </div>
          </div>

          <form onSubmit={handleFormSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Metric Type <span className="text-rose-500">*</span>
              </label>
              <select
                value={metricType}
                onChange={(e) => handleMetricChange(e.target.value as MetricType)}
                className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xs font-medium focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
              >
                {SUPPORTED_METRICS.map((m) => (
                  <option key={m} value={m}>
                    {formatMetricName(m)}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Measurement Value <span className="text-rose-500">*</span>
                </label>
                <input
                  type="number"
                  step="any"
                  required
                  value={metricValue}
                  onChange={(e) => setMetricValue(e.target.value)}
                  placeholder="e.g. 110"
                  className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Unit
                </label>
                <input
                  type="text"
                  value={unit}
                  onChange={(e) => setUnit(e.target.value)}
                  className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Activity Level
                </label>
                <select
                  value={activityLevel}
                  onChange={(e) => setActivityLevel(e.target.value as ActivityLevel)}
                  className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                >
                  <option value="UNKNOWN">Unknown / Null</option>
                  <option value="RESTING">Resting</option>
                  <option value="MODERATE">Moderate Activity</option>
                  <option value="INTENSE">Intense Activity</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Duration (mins)
                </label>
                <input
                  type="number"
                  min="0"
                  value={activityMinutes}
                  onChange={(e) => setActivityMinutes(e.target.value)}
                  placeholder="Optional"
                  className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                />
              </div>
            </div>

            <div className="pt-2">
              <Button type="submit" variant="teal" size="md" className="w-full" isLoading={isSubmitting}>
                Save Observation
              </Button>
            </div>
          </form>
        </div>

        {/* Right: Recent Observations List */}
        <div className="lg:col-span-7 bg-white rounded-2xl p-6 border border-slate-200 shadow-card">
          <div className="flex items-center justify-between pb-4 border-b border-slate-100 mb-4">
            <div>
              <h2 className="text-base font-bold text-slate-900">Recent Observations</h2>
              <p className="text-xs text-slate-400">
                Total recorded: {observations.length}
              </p>
            </div>
          </div>

          <ObservationList observations={observations} />
        </div>
      </div>
    </div>
  );
};

export default HealthDataPage;
