import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { HealthObservation } from '../../types/observation';
import { formatDate } from '../../utils/formatters';
import { LineChart as ChartIcon } from 'lucide-react';

interface ObservationTrendChartProps {
  observations: HealthObservation[];
  baselineValue?: number | null;
  metricName: string;
  unit: string;
}

export const ObservationTrendChart: React.FC<ObservationTrendChartProps> = ({
  observations,
  baselineValue,
  metricName,
  unit,
}) => {
  // Sort real observations chronologically
  const chartData = [...observations]
    .sort((a, b) => new Date(a.loggedAt).getTime() - new Date(b.loggedAt).getTime())
    .map((obs) => ({
      date: formatDate(obs.loggedAt),
      rawDate: obs.loggedAt,
      value: obs.metricValue,
      activity: obs.activityLevel,
    }));

  if (chartData.length < 2) {
    return (
      <div className="h-64 flex flex-col items-center justify-center bg-slate-50/70 rounded-2xl border border-dashed border-slate-200 text-center p-6">
        <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 mb-3">
          <ChartIcon className="w-5 h-5" />
        </div>
        <p className="text-sm font-semibold text-slate-700 mb-1">
          Insufficient Observations for Charting
        </p>
        <p className="text-xs text-slate-500 max-w-xs">
          At least 2 real {metricName} measurements are required to render an authentic trend chart. Currently recorded: {chartData.length}.
        </p>
      </div>
    );
  }

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: '#64748b' }}
            tickLine={false}
            axisLine={{ stroke: '#e2e8f0' }}
          />
          <YAxis
            domain={['auto', 'auto']}
            tick={{ fontSize: 11, fill: '#64748b' }}
            tickLine={false}
            axisLine={{ stroke: '#e2e8f0' }}
            unit={` ${unit}`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0f172a',
              borderRadius: '12px',
              border: 'none',
              color: '#ffffff',
              fontSize: '12px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            }}
            formatter={(value: any) => [`${value} ${unit}`, metricName]}
            labelFormatter={(label) => `Logged on ${label}`}
          />
          {baselineValue != null && (
            <ReferenceLine
              y={baselineValue}
              stroke="#0d9488"
              strokeDasharray="4 4"
              label={{
                value: `Baseline: ${baselineValue.toFixed(1)} ${unit}`,
                position: 'top',
                fill: '#0d9488',
                fontSize: 10,
              }}
            />
          )}
          <Line
            type="linear"
            dataKey="value"
            stroke="#2563eb"
            strokeWidth={3}
            dot={{ r: 4, fill: '#2563eb', stroke: '#ffffff', strokeWidth: 2 }}
            activeDot={{ r: 6, fill: '#0d9488' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ObservationTrendChart;
