import React from 'react';

export const CardSkeleton: React.FC<{ count?: number; className?: string }> = ({
  count = 1,
  className = '',
}) => {
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className={`bg-white rounded-2xl border border-slate-200/80 p-6 animate-pulse ${className}`}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="h-4 bg-slate-200 rounded w-1/3"></div>
            <div className="w-8 h-8 bg-slate-100 rounded-lg"></div>
          </div>
          <div className="h-8 bg-slate-200 rounded w-1/2 mb-2"></div>
          <div className="h-3 bg-slate-100 rounded w-2/3"></div>
        </div>
      ))}
    </>
  );
};

export const TableSkeleton: React.FC<{ rows?: number }> = ({ rows = 4 }) => {
  return (
    <div className="w-full bg-white rounded-2xl border border-slate-200 p-4 animate-pulse">
      <div className="h-5 bg-slate-200 rounded w-1/4 mb-4"></div>
      <div className="space-y-3">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="flex space-x-4">
            <div className="h-4 bg-slate-100 rounded flex-1"></div>
            <div className="h-4 bg-slate-100 rounded flex-1"></div>
            <div className="h-4 bg-slate-100 rounded flex-1"></div>
            <div className="h-4 bg-slate-100 rounded w-16"></div>
          </div>
        ))}
      </div>
    </div>
  );
};
