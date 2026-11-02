import React from 'react';
import { ChatMessage } from '../../types/chatbot';
import { AlertCircle, Bot, User, CheckCircle2, AlertTriangle, ShieldAlert, Heart } from 'lucide-react';
import Badge from '../common/Badge';

interface MessageBubbleProps {
  message: ChatMessage;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.sender === 'user';
  const isUrgent = message.safetyStatus === 'URGENT_MEDICAL_EVALUATION';
  const isNormal = message.safetyStatus === 'NORMAL_CONVERSATION' || message.safetyStatus === 'SAFE';

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start my-3`}>
      {/* Avatar */}
      <div
        className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 shadow-sm ${
          isUser
            ? 'bg-blue-600 text-white'
            : isUrgent
            ? 'bg-rose-600 text-white'
            : 'bg-teal-600 text-white'
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : isUrgent ? <ShieldAlert className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
      </div>

      {/* Bubble Container */}
      <div className={`max-w-[85%] sm:max-w-[75%] space-y-2`}>
        {/* Red Flag Emergency Alert Callout */}
        {isUrgent && (
          <div className="p-3.5 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs shadow-sm">
            <div className="flex items-center gap-1.5 font-bold text-rose-900 mb-1">
              <AlertCircle className="w-4 h-4 text-rose-600" />
              <span>Urgent Clinical Safety Protocol Triggered</span>
            </div>
            <p className="leading-relaxed">
              The reported symptoms indicate potential red-flag medical warning signs requiring prompt medical evaluation. Please contact emergency services (such as 911 or your local emergency hospital) or consult an emergency healthcare professional immediately.
            </p>
          </div>
        )}

        <div
          className={`p-4 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? 'bg-blue-600 text-white rounded-tr-sm'
              : 'bg-white border border-slate-200 text-slate-800 shadow-card rounded-tl-sm'
          }`}
        >
          <div className="whitespace-pre-wrap">{message.text}</div>

          {/* Structured clinical entities preview (Current Turn) */}
          {(() => {
            const displayExtraction = message.currentTurnExtraction !== undefined
              ? message.currentTurnExtraction
              : message.structuredData;

            const hasSymptoms = !!(displayExtraction?.symptoms && displayExtraction.symptoms.length > 0);
            const hasNegatives = !!(displayExtraction?.negative_symptoms && displayExtraction.negative_symptoms.length > 0);
            const hasMeasurements = !!(displayExtraction?.measurements && displayExtraction.measurements.length > 0);

            if (!hasSymptoms && !hasNegatives && !hasMeasurements) return null;

            return (
              <>
                {hasSymptoms && (
                  <div className="mt-3 pt-3 border-t border-slate-100 flex flex-wrap gap-1.5">
                    <span className="text-[11px] font-semibold text-slate-400 w-full block">
                      Extracted Symptoms (Current Turn):
                    </span>
                    {displayExtraction!.symptoms!.map((s: any, idx: number) => {
                      const symptomName = s.name || s.symptom_name;
                      return (
                        <Badge key={idx} variant="teal" size="sm">
                          {symptomName} {s.severity ? `(${s.severity})` : ''}
                        </Badge>
                      );
                    })}
                  </div>
                )}

                {hasNegatives && (
                  <div className="mt-2 pt-2 border-t border-slate-100 flex flex-wrap gap-1.5">
                    <span className="text-[11px] font-semibold text-slate-400 w-full block">
                      Reported Absent / Denied:
                    </span>
                    {displayExtraction!.negative_symptoms!.map((n: string, idx: number) => (
                      <Badge key={idx} variant="default" size="sm">
                        {n}
                      </Badge>
                    ))}
                  </div>
                )}

                {hasMeasurements && (
                  <div className="mt-2 pt-2 border-t border-slate-100 flex flex-wrap gap-1.5">
                    <span className="text-[11px] font-semibold text-slate-400 w-full block">
                      Extracted Measurements:
                    </span>
                    {displayExtraction!.measurements!.map((m: any, idx: number) => (
                      <Badge key={idx} variant="info" size="sm">
                        {m.name}: {m.value} {m.unit || ''}
                      </Badge>
                    ))}
                  </div>
                )}
              </>
            );
          })()}
        </div>

        {/* Compact Heart Disease Assessment Card */}
        {(() => {
          const handoff = message.assessmentHandoff;
          if (!handoff || !handoff.assessment_result) return null;
          const res = handoff.assessment_result;
          const isHeart = handoff.disease_candidates?.includes('heart_disease') || res.disease === 'HEART_DISEASE';
          if (!isHeart) return null;

          const prob = typeof res.probability === 'number' ? res.probability : 0;
          const probPct = (prob * 100).toFixed(1) + '%';
          const threshold = typeof res.threshold === 'number' ? res.threshold : 0.31;
          const thresholdPct = Math.round(threshold * 100) + '%';
          const isPositive = res.prediction === 1;
          const posContribs = (res.positiveContributors || []).slice(0, 2);
          const negContribs = (res.negativeContributors || []).slice(0, 2);

          return (
            <div className="p-3.5 rounded-2xl bg-white border border-slate-200 shadow-card space-y-2.5 text-xs text-slate-700">
              <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                <div className="flex items-center gap-1.5 font-semibold text-slate-800">
                  <Heart className="w-4 h-4 text-rose-500 fill-rose-500/20" />
                  <span>Heart Disease Assessment</span>
                </div>
                <Badge variant={isPositive ? 'warning' : 'success'} size="sm">
                  {isPositive ? 'Positive (Elevated)' : 'Negative (Below Cutoff)'}
                </Badge>
              </div>

              <div className="grid grid-cols-2 gap-2 text-center bg-slate-50 p-2 rounded-xl border border-slate-100">
                <div>
                  <div className="text-[10px] text-slate-400 font-medium">Model Probability</div>
                  <div className="text-sm font-bold text-slate-800">{probPct}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-400 font-medium">Decision Threshold</div>
                  <div className="text-sm font-semibold text-slate-600">{thresholdPct}</div>
                </div>
              </div>

              {(posContribs.length > 0 || negContribs.length > 0) && (
                <div className="space-y-1">
                  <span className="text-[10px] font-semibold text-slate-400 block">Top Model Contributors (SHAP):</span>
                  <div className="space-y-0.5 text-[11px]">
                    {posContribs.map((c: any, i: number) => (
                      <div key={`pos-${i}`} className="flex items-center justify-between text-rose-700">
                        <span className="truncate max-w-[170px]">+{c.displayName || c.feature}</span>
                        <span className="font-mono text-[10px]">+{Number(c.contribution).toFixed(3)}</span>
                      </div>
                    ))}
                    {negContribs.map((c: any, i: number) => (
                      <div key={`neg-${i}`} className="flex items-center justify-between text-emerald-700">
                        <span className="truncate max-w-[170px]">{c.displayName || c.feature}</span>
                        <span className="font-mono text-[10px]">{Number(c.contribution).toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="pt-1.5 border-t border-slate-100 text-[10px] text-slate-400 leading-tight">
                Research-based screening output — not a diagnosis.
              </div>
            </div>
          );
        })()}

        {/* Timestamp & Metadata */}
        <div
          className={`flex items-center gap-2 text-[10px] text-slate-400 px-1 ${
            isUser ? 'justify-end' : 'justify-start'
          }`}
        >
          <span>{message.timestamp}</span>
          {message.safetyStatus && (
            <span
              className={`font-semibold ${
                isNormal
                  ? 'text-emerald-600'
                  : isUrgent
                  ? 'text-rose-600'
                  : 'text-amber-600'
              }`}
            >
              • {message.safetyStatus.replace(/_/g, ' ')}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
