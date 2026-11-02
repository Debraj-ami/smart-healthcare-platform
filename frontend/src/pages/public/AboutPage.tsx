import React from 'react';
import { ArrowDown, Brain, Database, HeartPulse, LineChart, Shield, UserCheck, Bot } from 'lucide-react';
import DisclaimerAlert from '../../components/common/DisclaimerAlert';

export const AboutPage: React.FC = () => {
  const architectureSteps = [
    {
      title: '1. Patient Biometrics & Observations',
      icon: HeartPulse,
      color: 'text-teal-600 bg-teal-50',
      description:
        'Patient biometric measurements (glucose, heart rate, blood pressure, etc.) logged with temporal and activity context.',
    },
    {
      title: '2. Personal Baseline Engine',
      icon: LineChart,
      color: 'text-blue-600 bg-blue-50',
      description:
        'Dynamic computation of patient-specific running baseline means and standard deviations, avoiding population-level bias.',
    },
    {
      title: '3. Context-Aware Trend Analysis',
      icon: Database,
      color: 'text-cyan-600 bg-cyan-50',
      description:
        'Statistical persistence tracking, inflection detection, signed and absolute deviation scores with immutable snapshot preservation.',
    },
    {
      title: '4. Multi-Disease Risk Screening',
      icon: Brain,
      color: 'text-purple-600 bg-purple-50',
      description:
        'Disease-specific predictive modeling evaluating calibrated risk probabilities with clinical feature rankings.',
    },
    {
      title: '5. Conversational Safety & Clinical Triage',
      icon: Bot,
      color: 'text-amber-600 bg-amber-50',
      description:
        'FastAPI NLP conversational agent performing clinical entity extraction, targeted follow-up queries, and emergency triage routing.',
    },
    {
      title: '6. Patient & Clinician Dashboard',
      icon: UserCheck,
      color: 'text-emerald-600 bg-emerald-50',
      description:
        'Transparent, explainable user interface facilitating proactive health reviews and longitudinal clinical discussions.',
    },
  ];

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-12">
      {/* Title */}
      <div className="text-center space-y-3">
        <span className="text-xs font-bold uppercase tracking-wider text-teal-600 bg-teal-50 px-3 py-1 rounded-full border border-teal-200">
          Architecture & Methodology
        </span>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          About Smart Healthcare Platform
        </h1>
        <p className="text-base text-slate-600 max-w-2xl mx-auto leading-relaxed">
          A principled, data-driven research architecture designed for longitudinal wellness monitoring and early disease risk screening.
        </p>
      </div>

      <DisclaimerAlert variant="info" />

      {/* Conceptual Flow Diagram */}
      <section className="bg-white rounded-3xl p-8 border border-slate-200 shadow-card space-y-8">
        <h2 className="text-xl font-bold text-slate-900 text-center">
          System Architecture Flow
        </h2>

        <div className="max-w-2xl mx-auto space-y-4">
          {architectureSteps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <React.Fragment key={idx}>
                <div className="flex items-start gap-4 p-5 rounded-2xl bg-slate-50 border border-slate-200 hover:border-slate-300 transition-all">
                  <div
                    className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${step.color}`}
                  >
                    <Icon className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-slate-800">{step.title}</h3>
                    <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                      {step.description}
                    </p>
                  </div>
                </div>

                {idx < architectureSteps.length - 1 && (
                  <div className="flex justify-center text-slate-300">
                    <ArrowDown className="w-5 h-5 animate-bounce" />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </section>

      {/* Core Principles */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-card space-y-2.5">
          <div className="w-9 h-9 rounded-lg bg-teal-50 text-teal-600 flex items-center justify-center font-bold">
            <Shield className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-slate-800">Clear Clinical Boundaries</h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            The platform explicitly separates descriptive statistical observation and personalized monitoring from medical diagnosis. All assessments are probabilistic decision-support tools meant to empower clinical consultation.
          </p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-card space-y-2.5">
          <div className="w-9 h-9 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center font-bold">
            <LineChart className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-slate-800">Immutable Historical Snapshots</h3>
          <p className="text-xs text-slate-500 leading-relaxed">
            Every trend analysis preserves the exact context, previous observation delta, and activity parameters that existed at creation time. Historical evaluations are never retroactively overwritten by future measurements.
          </p>
        </div>
      </section>
    </div>
  );
};

export default AboutPage;
