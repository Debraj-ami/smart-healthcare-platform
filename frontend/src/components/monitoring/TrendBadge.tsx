import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { TrendDirection } from '../../types/trend';

export interface TrendBadgeProps {
  direction?: TrendDirection | string;
  persistence?: number | null;
  showPrefix?: boolean;
  className?: string;
}

export const TrendBadge: React.FC<TrendBadgeProps> = ({
  direction = 'STABLE',
  persistence,
  showPrefix = false,
  className = '',
}) => {
  const config = {
    INCREASING: {
      bg: 'bg-amber-50',
      text: 'text-amber-700',
      border: 'border-amber-200',
      icon: TrendingUp,
      label: 'Increasing',
    },
    DECREASING: {
      bg: 'bg-blue-50',
      text: 'text-blue-700',
      border: 'border-blue-200',
      icon: TrendingDown,
      label: 'Decreasing',
    },
    STABLE: {
      bg: 'bg-emerald-50',
      text: 'text-emerald-700',
      border: 'border-emerald-200',
      icon: Minus,
      label: 'Stable',
    },
  }[direction as TrendDirection] || {
    bg: 'bg-slate-50',
    text: 'text-slate-700',
    border: 'border-slate-200',
    icon: Minus,
    label: direction,
  };

  const Icon = config.icon;
  const labelText = showPrefix ? `Trend: ${config.label}` : config.label;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${config.bg} ${config.text} ${config.border} ${className}`}
    >
      <Icon className="w-3.5 h-3.5" />
      <span>{labelText}</span>
      {persistence && persistence > 1 && !showPrefix && (
        <span className="opacity-75 font-normal ml-0.5">({persistence} consecutive observations)</span>
      )}
    </span>
  );
};

export default TrendBadge;
