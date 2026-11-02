import React from 'react';
import { TrendAnalysis } from '../../types/trend';
import { Activity, BarChart2, Calendar, Clock, Gauge, History, Zap } from 'lucide-react';
import TrendBadge from './TrendBadge';

interface ContextStatsGridProps {
  trend: TrendAnalysis;
  unit: string;
}

export const ContextStatsGrid: React.FC<ContextStatsGridProps> = ({ trend, unit }) => {
  const hc = trend.historicalContext;
  const tc = trend.temporalContext;
  const ac = trend.activityContext;

  return (
    <div className="space-y-4">
      {/* Primary KPI Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-4 rounded-xl bg-white border border-slate-200/80 shadow-card">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block mb-1">
            Current Value
          </span>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-slate-900">{trend.currentValue}</span>
            <span className="text-xs text-slate-500 font-medium">{unit}</span>
          </div>
          {trend.valueChangeFromPrevious != null && (
            <span
              className={`text-xs font-semibold ${
                trend.valueChangeFromPrevious > 0
                  ? 'text-amber-600'
                  : trend.valueChangeFromPrevious < 0
                  ? 'text-blue-600'
                  : 'text-slate-500'
              }`}
            >
              {trend.valueChangeFromPrevious > 0 ? '+' : ''}
              {trend.valueChangeFromPrevious} from previous
            </span>
          )}
        </div>

        <div className="p-4 rounded-xl bg-white border border-slate-200/80 shadow-card">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block mb-1">
            Personal Baseline
          </span>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-teal-600">
              {trend.baselineValue != null ? trend.baselineValue.toFixed(1) : 'N/A'}
            </span>
            <span className="text-xs text-slate-500 font-medium">{unit}</span>
          </div>
          <span className="text-xs text-slate-400">
            {trend.variability != null ? `SD: ±${trend.variability.toFixed(1)}` : 'Calculated historical mean'}
          </span>
        </div>

        <div className="p-4 rounded-xl bg-white border border-slate-200/80 shadow-card">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block mb-1">
            Absolute Deviation
          </span>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-slate-800">
              {trend.absoluteDeviation != null ? trend.absoluteDeviation.toFixed(1) : 'N/A'}
            </span>
            <span className="text-xs text-slate-500 font-medium">{unit}</span>
          </div>
          <span className="text-xs text-slate-500">
            Signed: {trend.deviationFromBaseline != null ? `${trend.deviationFromBaseline > 0 ? '+' : ''}${trend.deviationFromBaseline.toFixed(1)} ${unit}` : 'N/A'}
          </span>
        </div>

        <div className="p-4 rounded-xl bg-white border border-slate-200/80 shadow-card">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block mb-1">
            Trend Direction
          </span>
          <div className="mt-1">
            <TrendBadge direction={trend.trendDirection} showPrefix={true} />
          </div>
          <span className="text-xs text-slate-500 mt-1.5 block font-medium">
            {trend.persistence ?? 1} consecutive observations
          </span>
        </div>
      </div>

      {/* Secondary Context Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Historical Context */}
        <div className="p-4 rounded-xl bg-slate-50/70 border border-slate-200">
          <div className="flex items-center gap-2 mb-3 text-xs font-bold text-slate-700 uppercase tracking-wider">
            <History className="w-4 h-4 text-teal-600" />
            <span>Historical Context</span>
          </div>
          {hc ? (
            <div className="space-y-1.5 text-xs text-slate-600">
              <div className="flex justify-between">
                <span>Total Observations:</span>
                <span className="font-semibold text-slate-800">{hc.observationCount}</span>
              </div>
              <div className="flex justify-between">
                <span>Historical Mean:</span>
                <span className="font-semibold text-slate-800">{hc.historicalMean.toFixed(1)} {unit}</span>
              </div>
              <div className="flex justify-between">
                <span>Recent Mean (Window):</span>
                <span className="font-semibold text-slate-800">{hc.recentMean.toFixed(1)} {unit}</span>
              </div>
              <div className="flex justify-between">
                <span>Historical Range:</span>
                <span className="font-semibold text-slate-800">{hc.historicalMinimum} – {hc.historicalMaximum} {unit}</span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">Contextual statistics pending additional observations.</p>
          )}
        </div>

        {/* Temporal Context */}
        <div className="p-4 rounded-xl bg-slate-50/70 border border-slate-200">
          <div className="flex items-center gap-2 mb-3 text-xs font-bold text-slate-700 uppercase tracking-wider">
            <Calendar className="w-4 h-4 text-blue-600" />
            <span>Temporal Context</span>
          </div>
          {tc ? (
            <div className="space-y-1.5 text-xs text-slate-600">
              <div className="flex justify-between">
                <span>Observation Date:</span>
                <span className="font-semibold text-slate-800">{tc.observationDate}</span>
              </div>
              <div className="flex justify-between">
                <span>Time of Day:</span>
                <span className="font-semibold text-slate-800">{tc.timeOfDay} ({tc.hourOfDay}:00)</span>
              </div>
              <div className="flex justify-between">
                <span>Day of Week:</span>
                <span className="font-semibold text-slate-800">{tc.dayOfWeek}</span>
              </div>
              <div className="flex justify-between">
                <span>Timestamp:</span>
                <span className="font-semibold text-slate-800">{tc.observationTime}</span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">Temporal context recorded with observation timestamp.</p>
          )}
        </div>

        {/* Activity Context */}
        <div className="p-4 rounded-xl bg-slate-50/70 border border-slate-200">
          <div className="flex items-center gap-2 mb-3 text-xs font-bold text-slate-700 uppercase tracking-wider">
            <Zap className="w-4 h-4 text-amber-600" />
            <span>Activity Context</span>
          </div>
          {ac ? (
            <div className="space-y-1.5 text-xs text-slate-600">
              <div className="flex justify-between">
                <span>Activity Level:</span>
                <span className="font-semibold text-slate-800">{ac.activityLevel || 'UNKNOWN'}</span>
              </div>
              <div className="flex justify-between">
                <span>Activity Duration:</span>
                <span className="font-semibold text-slate-800">
                  {ac.activityMinutes != null ? `${ac.activityMinutes} minutes` : 'Not specified'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-2">
                Preserved isolation: Activity state represents conditions at the exact time this trend was analyzed.
              </p>
            </div>
          ) : (
            <p className="text-xs text-slate-400">No activity context recorded for this observation.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ContextStatsGrid;
