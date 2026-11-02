import React, { useEffect } from 'react';
import {
  X,
  Heart,
  Clock,
  Activity,
  TrendingUp,
  TrendingDown,
  Info,
  Cpu,
  User,
  ShieldCheck,
  AlertCircle,
  CheckCircle2,
} from 'lucide-react';
import { DiseaseAssessment } from '../../types/assessment';
import { parseHeartAssessment, FormattedHeartInput } from '../../utils/heartAssessmentFormatter';
import { formatDateTime } from '../../utils/formatters';
import Button from '../common/Button';
import FloatingAIAvatar from '../chatbot/FloatingAIAvatar';
import { useChatAssistant } from '../../context/ChatAssistantContext';

interface HeartAssessmentDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  assessment: DiseaseAssessment | null;
}

export const HeartAssessmentDetailModal: React.FC<HeartAssessmentDetailModalProps> = ({
  isOpen,
  onClose,
  assessment,
}) => {
  const chatAssistant = useChatAssistant();
  const assessmentId = assessment?.id;

  useEffect(() => {
    if (isOpen && assessmentId && chatAssistant) {
      chatAssistant.setPageContext({
        source: 'assessment',
        disease: 'HEART',
        assessmentId: assessmentId,
      });
      return () => {
        chatAssistant.setPageContext(null);
      };
    }
  }, [isOpen, assessmentId, chatAssistant]);

  if (!isOpen || !assessment) return null;

  const parsed = parseHeartAssessment(assessment);

  // Compute maximum absolute SHAP contribution for responsive bar scaling
  const maxAbsContrib = Math.max(
    0.1,
    ...parsed.allContributions.map((c) => Math.abs(c.contribution)),
    ...parsed.positiveContributors.map((c) => Math.abs(c.contribution)),
    ...parsed.negativeContributors.map((c) => Math.abs(c.contribution))
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-slate-900/60 backdrop-blur-sm overflow-y-auto"
      role="dialog"
      aria-modal="true"
      aria-labelledby="heart-detail-title"
    >
      <div className="bg-white rounded-2xl max-w-3xl w-full max-h-[92vh] flex flex-col shadow-2xl border border-slate-200 animate-in fade-in zoom-in-95 duration-200">
        {/* Modal Header */}
        <div className="p-4 sm:p-6 border-b border-slate-100 flex items-center justify-between bg-gradient-to-r from-rose-50/50 via-white to-slate-50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-rose-100 text-rose-600 flex items-center justify-center flex-shrink-0 shadow-sm">
              <Heart className="w-5 h-5 fill-rose-500/20" />
            </div>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h2 id="heart-detail-title" className="text-base sm:text-lg font-bold text-slate-900">
                  Heart Assessment Detail
                </h2>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 font-semibold border border-slate-200">
                  #{parsed.id}
                </span>
                <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200">
                  Screening Output
                </span>
              </div>
              <p className="text-xs text-slate-500 flex items-center gap-1.5 mt-0.5">
                <Clock className="w-3.5 h-3.5 text-slate-400" />
                <span>Assessed: {formatDateTime(parsed.assessedAt)}</span>
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-600 rounded-xl hover:bg-slate-100 transition-colors"
            aria-label="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-4 sm:p-6 overflow-y-auto space-y-6 text-slate-800">
          {/* Section 1: Model Prediction & Probability Overview */}
          <div
            className={`p-4 sm:p-5 rounded-2xl border transition-all ${
              parsed.isPositive
                ? 'bg-rose-50/60 border-rose-200 text-rose-950'
                : 'bg-emerald-50/60 border-emerald-200 text-emerald-950'
            }`}
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-black/5">
              <div className="space-y-0.5">
                <span className="text-xs font-semibold uppercase tracking-wider opacity-75">
                  Model Screening Result
                </span>
                <div className="flex items-center gap-2">
                  {parsed.isPositive ? (
                    <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0" />
                  ) : (
                    <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0" />
                  )}
                  <h3 className="text-lg font-extrabold tracking-tight">
                    Model Prediction: {parsed.isPositive ? 'Positive (Elevated Score)' : 'Negative (Below Threshold)'}
                  </h3>
                </div>
              </div>

              <div className="sm:text-right">
                <span className="text-xs opacity-75 block font-medium">Model-estimated probability</span>
                <span className="text-2xl font-black font-mono tracking-tight">
                  {parsed.probabilityPercentage}
                </span>
              </div>
            </div>

            {/* Threshold & Probability Bar */}
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold">
                <span>Model Output Distribution</span>
                <span className="font-mono">
                  Probability: {parsed.probability.toFixed(4)} · Decision Threshold: t = {parsed.threshold.toFixed(2)}
                </span>
              </div>

              <div className="relative h-4 bg-slate-200/80 rounded-full overflow-hidden shadow-inner">
                {/* Decision Threshold Vertical Marker */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-slate-900 z-10"
                  style={{ left: `${parsed.threshold * 100}%` }}
                  title={`Decision threshold t = ${parsed.threshold.toFixed(2)}`}
                />
                {/* Probability Bar */}
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    parsed.isPositive ? 'bg-rose-500' : 'bg-emerald-500'
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, parsed.probability * 100))}%` }}
                />
              </div>

              <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1">
                <span>0.00 (Minimum)</span>
                <span className="font-semibold text-slate-700">
                  Cutoff: {(parsed.threshold * 100).toFixed(1)}%
                </span>
                <span>1.00 (Maximum)</span>
              </div>
            </div>

            {/* Threshold Educational Callout */}
            <div className="mt-3 p-3 bg-white/70 backdrop-blur-xs rounded-xl border border-black/5 text-xs text-slate-600 leading-relaxed flex items-start gap-2">
              <Info className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
              <p>
                <strong>Understanding the Decision Threshold:</strong> The probability is the model's estimated output for this assessment. The threshold is the operating cutoff used to convert that probability into the model's binary prediction. It does not represent a clinical diagnostic cutoff or guarantee individual disease presence or absence.
              </p>
            </div>
          </div>

          {/* Section 2: Patient Context & Personalization State */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-200">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-800">
                <User className="w-4 h-4 text-rose-500" />
                <span>Patient Assessment Context</span>
              </div>
              <span className="text-[11px] font-mono text-slate-500">
                Patient #{parsed.patientId}
              </span>
            </div>

            {parsed.patientContext ? (
              <div className="space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                  <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                    <span className="text-slate-400 block text-[11px]">Previous Assessments</span>
                    <span className="font-bold text-slate-800 text-sm">
                      {parsed.patientContext.previousAssessmentCount} recorded
                    </span>
                  </div>
                  <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                    <span className="text-slate-400 block text-[11px]">Previous Output Probability</span>
                    <span className="font-bold text-slate-800 text-sm">
                      {parsed.patientContext.lastProbability !== null
                        ? `${(parsed.patientContext.lastProbability * 100).toFixed(1)}%`
                        : 'None (Initial screening)'}
                    </span>
                  </div>
                  <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                    <span className="text-slate-400 block text-[11px]">Prior Assessment Time</span>
                    <span className="font-medium text-slate-700">
                      {parsed.patientContext.lastAssessedAt
                        ? formatDateTime(parsed.patientContext.lastAssessedAt)
                        : 'N/A'}
                    </span>
                  </div>
                  <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                    <span className="text-slate-400 block text-[11px]">Personal Baselines</span>
                    <span className="font-medium text-slate-700">
                      {parsed.patientContext.hasEstablishedBaselines
                        ? parsed.patientContext.establishedBaselineMetrics.join(', ') || 'Active'
                        : 'None established'}
                    </span>
                  </div>
                </div>

                {parsed.patientContext.contextNotes && (
                  <p className="text-xs text-slate-600 bg-white p-2.5 rounded-lg border border-slate-200 italic leading-relaxed">
                    Context Note: {parsed.patientContext.contextNotes}
                  </p>
                )}

                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Note: Longitudinal probabilities represent previous model-estimated assessment outputs rather than disease progression or vital tracking.
                </p>
              </div>
            ) : (
              <div className="text-xs text-slate-500 py-1">
                <p>Standard screening record for Patient #{parsed.patientId}. No additional longitudinal notes stored.</p>
              </div>
            )}
          </div>

          {/* Section 3: Assessment Inputs (Reconstructed 13 Features) */}
          <div className="space-y-3">
            <div className="flex items-center justify-between pb-1 border-b border-slate-200">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-teal-600" />
                <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                  Submitted Clinical Features (13 Inputs)
                </h3>
              </div>
              <span className="text-[11px] text-slate-500">
                Cleveland Protocol Specification
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
              {parsed.inputs.map((item: FormattedHeartInput) => (
                <div
                  key={item.name}
                  className="p-2.5 bg-slate-50/70 hover:bg-slate-50 rounded-xl border border-slate-200/80 transition-colors"
                >
                  <div className="text-[11px] font-medium text-slate-500 truncate" title={item.label}>
                    {item.label}
                  </div>
                  <div className="text-xs font-bold text-slate-900 mt-0.5 flex items-center justify-between">
                    <span>{item.displayValue}</span>
                    {item.rawValue !== null && item.rawValue !== undefined && item.rawValue !== '' ? (
                      <span className="text-[10px] font-mono text-slate-400 font-normal">
                        ({String(item.rawValue)})
                      </span>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Section 4: SHAP Model-Output Explanations */}
          <div className="space-y-3">
            <div className="flex items-center justify-between pb-1 border-b border-slate-200">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-violet-600" />
                <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                  Model Output Attribution (SHAP Explanations)
                </h3>
              </div>
              {parsed.explanationAvailable && (
                <span className="text-[11px] font-mono text-slate-500">
                  Space: {parsed.explanationOutputSpace} · Base Value: {parsed.shapBaseValue !== null ? parsed.shapBaseValue.toFixed(4) : 'N/A'}
                </span>
              )}
            </div>

            <p className="text-xs text-slate-500 leading-relaxed">
              These values show how individual model features contributed to this model's output for this assessment. SHAP attributions represent statistical model contributions, not biological causation, clinical diagnoses, or proof of disease.
            </p>

            {parsed.explanationAvailable ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Positive (Risk-Increasing) Contributors */}
                <div className="space-y-2">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-rose-700 pb-1.5 border-b border-rose-200">
                    <TrendingUp className="w-4 h-4 text-rose-600" />
                    <span>Positive Contributors ({parsed.positiveContributors.length})</span>
                  </div>

                  {parsed.positiveContributors.length === 0 ? (
                    <p className="text-xs text-slate-400 py-4 text-center italic bg-slate-50 rounded-xl border border-dashed border-slate-200">
                      No positive contributors for this assessment.
                    </p>
                  ) : (
                    parsed.positiveContributors.map((c, i) => {
                      const pct = Math.min(100, Math.max(5, (Math.abs(c.contribution) / maxAbsContrib) * 100));
                      const isCategorical = c.transformedFeature?.startsWith('cat__');
                      return (
                        <div
                          key={`pos-${i}`}
                          className="p-2.5 bg-rose-50/40 rounded-xl border border-rose-100 space-y-1.5"
                        >
                          <div className="flex items-start justify-between gap-2 text-xs">
                            <div>
                              <span className="font-semibold text-slate-800 block">
                                {c.displayName || c.feature}
                              </span>
                              {isCategorical && (
                                <span className="text-[10px] text-slate-500">
                                  Encoded category: <span className="font-medium text-slate-700">{c.transformedFeature.replace(/^cat__/, '').replace(/_/g, ' ')}</span>
                                </span>
                              )}
                            </div>
                            <span className="font-mono text-xs font-bold text-rose-700 flex-shrink-0">
                              +{c.contribution.toFixed(4)}
                            </span>
                          </div>

                          <div className="h-1.5 bg-rose-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-rose-500 rounded-full"
                              style={{ width: `${pct}%` }}
                            />
                          </div>

                          <div className="text-[10px] text-slate-500 flex justify-between items-center">
                            <span>Contributed positively to model output</span>
                            {c.isCategorical ? (
                              <span className="font-mono">
                                {c.isActive ? (
                                  <span className="text-emerald-700 font-semibold bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                                    Active (1)
                                  </span>
                                ) : (
                                  <span className="text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">
                                    Inactive (0)
                                  </span>
                                )}
                              </span>
                            ) : (
                              c.rawValue !== undefined && c.rawValue !== null && (
                                <span className="font-mono">Input: {String(c.rawValue)}</span>
                              )
                            )}
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>

                {/* Negative (Risk-Decreasing) Contributors */}
                <div className="space-y-2">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-700 pb-1.5 border-b border-emerald-200">
                    <TrendingDown className="w-4 h-4 text-emerald-600" />
                    <span>Negative Contributors ({parsed.negativeContributors.length})</span>
                  </div>

                  {parsed.negativeContributors.length === 0 ? (
                    <p className="text-xs text-slate-400 py-4 text-center italic bg-slate-50 rounded-xl border border-dashed border-slate-200">
                      No negative contributors for this assessment.
                    </p>
                  ) : (
                    parsed.negativeContributors.map((c, i) => {
                      const pct = Math.min(100, Math.max(5, (Math.abs(c.contribution) / maxAbsContrib) * 100));
                      const isCategorical = c.transformedFeature?.startsWith('cat__');
                      return (
                        <div
                          key={`neg-${i}`}
                          className="p-2.5 bg-emerald-50/40 rounded-xl border border-emerald-100 space-y-1.5"
                        >
                          <div className="flex items-start justify-between gap-2 text-xs">
                            <div>
                              <span className="font-semibold text-slate-800 block">
                                {c.displayName || c.feature}
                              </span>
                              {isCategorical && (
                                <span className="text-[10px] text-slate-500">
                                  Encoded category: <span className="font-medium text-slate-700">{c.transformedFeature.replace(/^cat__/, '').replace(/_/g, ' ')}</span>
                                </span>
                              )}
                            </div>
                            <span className="font-mono text-xs font-bold text-emerald-700 flex-shrink-0">
                              {c.contribution.toFixed(4)}
                            </span>
                          </div>

                          <div className="h-1.5 bg-emerald-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-emerald-500 rounded-full"
                              style={{ width: `${pct}%` }}
                            />
                          </div>

                          <div className="text-[10px] text-slate-500 flex justify-between items-center">
                            <span>Contributed negatively to model output</span>
                            {c.isCategorical ? (
                              <span className="font-mono">
                                {c.isActive ? (
                                  <span className="text-emerald-700 font-semibold bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                                    Active (1)
                                  </span>
                                ) : (
                                  <span className="text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">
                                    Inactive (0)
                                  </span>
                                )}
                              </span>
                            ) : (
                              c.rawValue !== undefined && c.rawValue !== null && (
                                <span className="font-mono">Input: {String(c.rawValue)}</span>
                              )
                            )}
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            ) : (
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 flex items-start gap-2.5">
                <Info className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-slate-700">Explanation Notice:</span>
                  <p className="mt-0.5">An explanation is not available for this assessment.</p>
                  {parsed.explanationError && (
                    <p className="mt-1 text-[11px] text-slate-500 font-mono">Details: {parsed.explanationError}</p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Section 5: Model & Version Metadata */}
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl text-xs space-y-1.5 font-mono text-slate-600">
            <div className="flex items-center gap-1.5 text-slate-700 font-bold font-sans">
              <Cpu className="w-3.5 h-3.5 text-slate-500" />
              <span>Model & Serving Metadata</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 text-[11px] pt-1">
              <div>
                <span className="text-slate-400 block font-sans">Model Name:</span>
                <span className="text-slate-800 font-semibold">{parsed.modelName}</span>
              </div>
              <div>
                <span className="text-slate-400 block font-sans">Model Version:</span>
                <span className="text-slate-800">{parsed.modelVersion}</span>
              </div>
              <div>
                <span className="text-slate-400 block font-sans">Configuration ID:</span>
                <span className="text-slate-800">{parsed.configId}</span>
              </div>
              <div>
                <span className="text-slate-400 block font-sans">Output Space:</span>
                <span className="text-slate-800">{parsed.outputSpace}</span>
              </div>
              <div>
                <span className="text-slate-400 block font-sans">Calibration:</span>
                <span className="text-slate-800">{parsed.calibrationStatus}</span>
              </div>
              <div>
                <span className="text-slate-400 block font-sans">Database Record ID:</span>
                <span className="text-slate-800">#{parsed.id}</span>
              </div>
            </div>
          </div>

          {/* Section 6: Clinical & Research Safety Disclaimer */}
          <div className="p-3.5 rounded-xl bg-amber-50/60 border border-amber-200/80 text-amber-900 text-xs leading-relaxed flex items-start gap-2.5">
            <ShieldCheck className="w-4 h-4 text-amber-700 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-amber-800">Research & Screening Safety Notice:</span>
              <p className="mt-0.5">
                This assessment is a research-based statistical screening and risk-estimation output. It does not constitute a medical diagnosis, clinical prognosis, or treatment recommendation. Always consult a qualified cardiologist or healthcare professional for cardiovascular evaluation and medical decision-making.
              </p>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 sm:p-5 border-t border-slate-100 flex items-center justify-between bg-slate-50 rounded-b-2xl">
          <div className="flex items-center gap-2.5">
            <FloatingAIAvatar
              position="inline"
              size="sm"
              context={{
                source: 'assessment',
                disease: 'HEART',
                assessmentId: parsed.id,
              }}
              tooltipText="Ask Smart Health AI about this assessment"
            />
            <div className="hidden sm:block">
              <span className="text-xs font-bold text-slate-800 block">
                Ask Smart Health AI
              </span>
              <span className="text-[11px] text-slate-500">
                Explore probabilities, SHAP factors & questions for your doctor
              </span>
            </div>
          </div>

          <Button variant="outline" size="sm" onClick={onClose}>
            Close Detail
          </Button>
        </div>
      </div>
    </div>
  );
};

export default HeartAssessmentDetailModal;
