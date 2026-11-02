import React from 'react';
import {
  Brain,
  TrendingUp,
  Activity,
  MessageSquare,
  Bell,
  History,
  Watch,
  Lock,
  Cpu,
  CheckCircle2,
  Clock,
} from 'lucide-react';
import Badge from '../../components/common/Badge';

export const FeaturesPage: React.FC = () => {
  const features = [
    {
      title: 'Disease Risk Screening',
      icon: Brain,
      badge: 'Active ML Models',
      badgeVariant: 'teal' as const,
      description:
        'Multi-condition screening for Diabetes, Heart Disease, Parkinson’s, and Liver Disease based on validated biometric features.',
    },
    {
      title: 'Personalized Health Baselines',
      icon: TrendingUp,
      badge: 'Verified Backend API',
      badgeVariant: 'success' as const,
      description:
        'Calculates patient-specific mean and standard deviation variability across observations, establishing your biometric reference.',
    },
    {
      title: 'Context-Aware Trend Analysis',
      icon: Activity,
      badge: 'Verified Backend API',
      badgeVariant: 'success' as const,
      description:
        'Analyzes trend direction, persistence, signed baseline deltas, absolute deviations, temporal context, and isolated activity states.',
    },
    {
      title: 'Biometric Observation Tracking',
      icon: Activity,
      badge: 'Verified Backend API',
      badgeVariant: 'success' as const,
      description:
        'Records blood glucose, heart rate, blood pressure, temperature, SpO2, and weight with timestamps and activity logs.',
    },
    {
      title: 'Conversational Clinical Assistant',
      icon: MessageSquare,
      badge: 'FastAPI Microservice',
      badgeVariant: 'teal' as const,
      description:
        'Clinical information extraction, symptom severity analysis, follow-up clarification, and red-flag emergency triage.',
    },
    {
      title: 'Automated Health Alerts',
      icon: Bell,
      badge: 'Verified Backend API',
      badgeVariant: 'success' as const,
      description:
        'Contextual alerts generated when observations deviate critically or sustained negative trends are confirmed.',
    },
    {
      title: 'Longitudinal Clinical History',
      icon: History,
      badge: 'Verified Backend API',
      badgeVariant: 'success' as const,
      description:
        'Comprehensive audit log of all observations, risk assessments, trend analyses, and notifications over time.',
    },
    {
      title: 'Wearable & IoT Integration',
      icon: Watch,
      badge: 'Roadmap (Phase 13)',
      badgeVariant: 'warning' as const,
      description:
        'Upcoming integration with continuous glucose monitors, smartwatches, and telemetry sensors for streaming vitals.',
    },
    {
      title: 'Privacy & Data Governance',
      icon: Lock,
      badge: 'Active Architecture',
      badgeVariant: 'info' as const,
      description:
        'Strict tenant-level patient isolation, parameterized queries, and defensive data protection standards.',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-12">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <span className="text-xs font-bold uppercase tracking-wider text-teal-600 bg-teal-50 px-3 py-1 rounded-full border border-teal-200">
          Capabilities & Roadmap
        </span>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          Platform Features & Capabilities
        </h1>
        <p className="text-base text-slate-600 leading-relaxed">
          Explore the active capabilities implemented in our backend and microservices, alongside transparent roadmaps for future enhancements.
        </p>
      </div>

      {/* Feature Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {features.map((feat, idx) => {
          const Icon = feat.icon;
          return (
            <div
              key={idx}
              className="bg-white rounded-2xl p-6 border border-slate-200 shadow-card hover:shadow-card-hover transition-all flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div className="w-10 h-10 rounded-xl bg-slate-50 border border-slate-100 flex items-center justify-center text-teal-600">
                    <Icon className="w-5 h-5" />
                  </div>
                  <Badge variant={feat.badgeVariant} size="sm">
                    {feat.badge}
                  </Badge>
                </div>

                <h3 className="text-base font-bold text-slate-800 mb-2">{feat.title}</h3>
                <p className="text-xs text-slate-500 leading-relaxed">{feat.description}</p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] font-medium text-slate-400 flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-teal-500" />
                <span>Production Spec Specification</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default FeaturesPage;
