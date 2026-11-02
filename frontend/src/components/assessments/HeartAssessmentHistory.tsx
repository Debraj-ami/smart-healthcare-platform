import React, { useState, useEffect, useCallback } from 'react';
import {
  Heart,
  Clock,
  ChevronRight,
  AlertCircle,
  RefreshCw,
  Sliders,
} from 'lucide-react';
import { DiseaseAssessment } from '../../types/assessment';
import heartPredictionService from '../../services/heartPredictionService';
import { parseHeartAssessment } from '../../utils/heartAssessmentFormatter';
import { formatDateTime } from '../../utils/formatters';
import Button from '../common/Button';
import Badge from '../common/Badge';

interface HeartAssessmentHistoryProps {
  patientId: number;
  onSelectAssessment: (assessment: DiseaseAssessment) => void;
  onStartNewAssessment?: () => void;
  className?: string;
  refreshTrigger?: number;
}

export const HeartAssessmentHistory: React.FC<HeartAssessmentHistoryProps> = ({
  patientId,
  onSelectAssessment,
  onStartNewAssessment,
  className = '',
  refreshTrigger = 0,
}) => {
  const [history, setHistory] = useState<DiseaseAssessment[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fetchHistory = useCallback(async () => {
    if (!patientId || isNaN(patientId)) {
      setErrorMessage('A valid authenticated patient context is required.');
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const data = await heartPredictionService.getPatientHeartHistory(patientId);
      // Sort reverse chronological by assessedAt descending
      const sorted = [...data].sort((a, b) => {
        const timeA = new Date(a.assessedAt).getTime();
        const timeB = new Date(b.assessedAt).getTime();
        return timeB - timeA;
      });
      setHistory(sorted);
    } catch (err: unknown) {
      console.error('Failed to load heart assessment history for patient #' + patientId, err);
      const msg = err instanceof Error ? err.message : 'Network error loading heart history.';
      setErrorMessage(`Failed to retrieve Heart assessment records (${msg}).`);
    } finally {
      setIsLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory, refreshTrigger]);

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <Heart className="w-5 h-5 text-rose-500 fill-rose-500/20" />
            <h2 className="text-base font-bold text-slate-900">
              Heart Disease Assessment History
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Historical cardiovascular ML screenings for Patient #{patientId}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={fetchHistory}
            disabled={isLoading}
            className="flex items-center gap-1.5"
            title="Refresh history"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </Button>

          {onStartNewAssessment && (
            <Button
              variant="danger"
              size="sm"
              onClick={onStartNewAssessment}
              className="flex items-center gap-1.5"
            >
              <Heart className="w-3.5 h-3.5" />
              <span>New Assessment</span>
            </Button>
          )}
        </div>
      </div>

      {/* Loading Skeleton */}
      {isLoading && (
        <div className="space-y-3 animate-pulse" aria-busy="true" aria-label="Loading heart assessments">
          {[1, 2, 3].map((n) => (
            <div
              key={n}
              className="p-4 bg-slate-50 rounded-2xl border border-slate-200/80 flex items-center justify-between gap-4"
            >
              <div className="space-y-2 flex-1">
                <div className="h-4 bg-slate-200 rounded w-1/3" />
                <div className="h-3 bg-slate-200 rounded w-1/4" />
              </div>
              <div className="h-8 bg-slate-200 rounded w-24" />
            </div>
          ))}
        </div>
      )}

      {/* Error State */}
      {!isLoading && errorMessage && (
        <div
          role="alert"
          className="p-5 rounded-2xl bg-rose-50/80 border border-rose-200 text-rose-900 space-y-3"
        >
          <div className="flex items-center gap-2 font-bold text-sm text-rose-800">
            <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0" />
            <span>Unable to Load Assessment History</span>
          </div>
          <p className="text-xs text-rose-700 leading-relaxed">{errorMessage}</p>
          <Button variant="outline" size="sm" onClick={fetchHistory} className="bg-white">
            <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
            Retry Connection
          </Button>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && !errorMessage && history.length === 0 && (
        <div className="py-12 px-6 text-center bg-slate-50/70 rounded-2xl border border-dashed border-slate-200 space-y-3">
          <div className="w-12 h-12 rounded-2xl bg-rose-50 text-rose-500 flex items-center justify-center mx-auto">
            <Heart className="w-6 h-6 fill-rose-500/10" />
          </div>
          <div className="space-y-1">
            <h3 className="text-sm font-bold text-slate-800">
              No Heart assessments have been recorded yet.
            </h3>
            <p className="text-xs text-slate-500 max-w-sm mx-auto leading-relaxed">
              When a Heart disease screening assessment is completed for Patient #{patientId}, authentic model outputs, decision cutoffs, and SHAP explanations will appear here.
            </p>
          </div>
          {onStartNewAssessment && (
            <div className="pt-2">
              <Button variant="danger" size="sm" onClick={onStartNewAssessment}>
                <Heart className="w-3.5 h-3.5 mr-1.5" />
                Start First Heart Assessment
              </Button>
            </div>
          )}
        </div>
      )}

      {/* History List */}
      {!isLoading && !errorMessage && history.length > 0 && (
        <div className="space-y-3" role="list" aria-label="Heart assessment records">
          {history.map((assessment) => {
            const parsed = parseHeartAssessment(assessment);
            return (
              <div
                key={assessment.id}
                role="listitem"
                tabIndex={0}
                onClick={() => onSelectAssessment(assessment)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onSelectAssessment(assessment);
                  }
                }}
                className="group p-4 bg-white hover:bg-slate-50/80 rounded-2xl border border-slate-200 hover:border-rose-200 shadow-xs hover:shadow-md transition-all cursor-pointer flex flex-col sm:flex-row sm:items-center justify-between gap-4"
              >
                <div className="flex items-start gap-3.5">
                  <div
                    className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5 ${
                      parsed.isPositive
                        ? 'bg-rose-50 text-rose-600'
                        : 'bg-emerald-50 text-emerald-600'
                    }`}
                  >
                    <Heart className="w-5 h-5 fill-current/15" />
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-bold text-slate-900 group-hover:text-rose-600 transition-colors">
                        Heart Screening #{parsed.id}
                      </span>
                      <Badge variant={parsed.isPositive ? 'danger' : 'success'}>
                        {parsed.isPositive ? 'Positive (Elevated)' : 'Negative (Below Cutoff)'}
                      </Badge>
                      <span className="text-[11px] font-mono text-slate-400">
                        {parsed.modelName} v{parsed.modelVersion}
                      </span>
                    </div>

                    <div className="flex items-center gap-3 text-xs text-slate-500 flex-wrap">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3.5 h-3.5 text-slate-400" />
                        {formatDateTime(parsed.assessedAt)}
                      </span>
                      <span className="flex items-center gap-1 font-mono text-slate-600">
                        <Sliders className="w-3 h-3 text-slate-400" />
                        Threshold: t = {parsed.threshold.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between sm:justify-end gap-4 border-t sm:border-t-0 pt-3 sm:pt-0 border-slate-100">
                  <div className="text-left sm:text-right">
                    <span className="text-[11px] text-slate-400 block font-medium">
                      Model-estimated probability
                    </span>
                    <span
                      className={`text-lg font-black font-mono tracking-tight ${
                        parsed.isPositive ? 'text-rose-600' : 'text-emerald-600'
                      }`}
                    >
                      {parsed.probabilityPercentage}
                    </span>
                  </div>

                  <div className="flex items-center text-xs font-semibold text-rose-600 group-hover:translate-x-0.5 transition-transform">
                    <span className="hidden sm:inline mr-1">View Details</span>
                    <ChevronRight className="w-4 h-4" />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default HeartAssessmentHistory;
