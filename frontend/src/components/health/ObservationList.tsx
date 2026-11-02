import React from 'react';
import { HealthObservation } from '../../types/observation';
import { formatMetricName, formatDateTime } from '../../utils/formatters';
import { Activity, Clock, Heart, Zap, Shield } from 'lucide-react';
import Badge from '../common/Badge';

interface ObservationListProps {
  observations: HealthObservation[];
}

export const ObservationList: React.FC<ObservationListProps> = ({ observations }) => {
  if (observations.length === 0) {
    return (
      <div className="p-8 text-center bg-slate-50/50 rounded-2xl border border-dashed border-slate-200">
        <Activity className="w-8 h-8 text-slate-300 mx-auto mb-2" />
        <p className="text-sm font-medium text-slate-600">No observations recorded yet.</p>
        <p className="text-xs text-slate-400 mt-1">
          Record your first health measurement using the form.
        </p>
      </div>
    );
  }

  // Sort descending by logged date
  const sorted = [...observations].sort(
    (a, b) => new Date(b.loggedAt).getTime() - new Date(a.loggedAt).getTime()
  );

  return (
    <div className="divide-y divide-slate-100">
      {sorted.map((obs) => (
        <div
          key={obs.id}
          className="py-3.5 px-4 flex items-center justify-between hover:bg-slate-50/80 transition-colors rounded-xl"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center font-bold text-xs flex-shrink-0">
              <Heart className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-slate-800">
                  {formatMetricName(obs.metricType)}
                </span>
                {obs.activityLevel && obs.activityLevel !== 'UNKNOWN' && (
                  <Badge variant="teal" size="sm">
                    {obs.activityLevel.toLowerCase()}
                    {obs.activityMinutes ? ` (${obs.activityMinutes}m)` : ''}
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
                <Clock className="w-3.5 h-3.5" />
                <span>{formatDateTime(obs.loggedAt)}</span>
              </div>
            </div>
          </div>

          <div className="text-right">
            <span className="text-base font-bold text-slate-900">{obs.metricValue}</span>
            <span className="text-xs text-slate-500 ml-1 font-medium">{obs.unit}</span>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ObservationList;
