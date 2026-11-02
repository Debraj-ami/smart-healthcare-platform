import React, { useState } from 'react';
import { X, Plus, AlertCircle } from 'lucide-react';
import { MetricType, ActivityLevel } from '../../types/observation';
import observationService from '../../services/observationService';
import { formatMetricName, getDefaultUnit } from '../../utils/formatters';
import Button from '../common/Button';

interface AddObservationModalProps {
  isOpen: boolean;
  onClose: () => void;
  patientId: number;
  onObservationAdded: () => void;
}

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

export const AddObservationModal: React.FC<AddObservationModalProps> = ({
  isOpen,
  onClose,
  patientId,
  onObservationAdded,
}) => {
  const [metricType, setMetricType] = useState<MetricType>('GLUCOSE');
  const [metricValue, setMetricValue] = useState<string>('');
  const [unit, setUnit] = useState<string>('mg/dL');
  const [activityLevel, setActivityLevel] = useState<ActivityLevel>('UNKNOWN');
  const [activityMinutes, setActivityMinutes] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleMetricChange = (type: MetricType) => {
    setMetricType(type);
    setUnit(getDefaultUnit(type));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const val = parseFloat(metricValue);
    if (isNaN(val)) {
      setError('Please enter a valid numeric measurement');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      await observationService.recordObservation({
        patientId,
        metricType,
        metricValue: val,
        unit,
        activityLevel,
        activityMinutes: activityMinutes ? parseInt(activityMinutes, 10) : null,
      });

      setMetricValue('');
      setActivityMinutes('');
      onObservationAdded();
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to record health observation');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-xl border border-slate-200 animate-in fade-in zoom-in-95 duration-150">
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-teal-50 text-teal-600 flex items-center justify-center">
              <Plus className="w-4 h-4" />
            </div>
            <h3 className="text-base font-bold text-slate-800">Add Health Observation</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="mt-4 p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="mt-4 space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1.5">
              Metric Type
            </label>
            <select
              value={metricType}
              onChange={(e) => handleMetricChange(e.target.value as MetricType)}
              className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
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
                Value <span className="text-rose-500">*</span>
              </label>
              <input
                type="number"
                step="any"
                required
                value={metricValue}
                onChange={(e) => setMetricValue(e.target.value)}
                placeholder="e.g. 110"
                className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
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
                className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
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
                className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
              >
                <option value="UNKNOWN">Unknown / Not Specified</option>
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
                placeholder="e.g. 30"
                className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
              />
            </div>
          </div>

          <div className="pt-3 border-t border-slate-100 flex items-center justify-end gap-2.5">
            <Button type="button" variant="ghost" size="sm" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="teal" size="sm" isLoading={isLoading}>
              Save Observation
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddObservationModal;
