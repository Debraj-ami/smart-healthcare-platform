import React from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  ArrowRight,
  ShieldCheck,
  TrendingUp,
  Brain,
  MessageSquare,
  Lock,
  HeartHandshake,
  CheckCircle2,
  Clock,
  ChevronRight,
  Zap,
} from 'lucide-react';
import Button from '../../components/common/Button';
import DisclaimerAlert from '../../components/common/DisclaimerAlert';

export const LandingPage: React.FC = () => {
  return (
    <div className="space-y-20 pb-20">
      {/* 1. HERO SECTION */}
      <section className="relative overflow-hidden pt-12 pb-16 lg:pt-20 lg:pb-24 bg-gradient-to-b from-teal-50/40 via-white to-slate-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
            {/* Left Content */}
            <div className="lg:col-span-7 space-y-6">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-teal-50 border border-teal-200/80 text-xs font-semibold text-teal-800">
                <span className="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></span>
                AI-Powered • Personalized • Context-Aware
              </div>

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-900 leading-[1.12]">
                Better Health.{' '}
                <span className="bg-gradient-to-r from-teal-600 via-cyan-600 to-blue-600 bg-clip-text text-transparent">
                  A Brighter Tomorrow.
                </span>
              </h1>

              <p className="text-lg text-slate-600 max-w-xl leading-relaxed">
                Smart Healthcare Platform integrates clinical observations, statistical personal baselines, context-aware trend analysis, and natural language assistant triage for proactive wellness monitoring.
              </p>

              <div className="flex flex-wrap items-center gap-4 pt-2">
                <Link to="/register">
                  <Button variant="teal" size="lg" rightIcon={<ArrowRight className="w-5 h-5" />}>
                    Get Started Free
                  </Button>
                </Link>
                <Link to="/dashboard">
                  <Button variant="outline" size="lg">
                    Explore Patient Portal
                  </Button>
                </Link>
              </div>

              {/* Value Badges */}
              <div className="pt-8 border-t border-slate-200/80 grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-teal-100/60 text-teal-700 flex items-center justify-center flex-shrink-0">
                    <Brain className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-800">AI-Powered Insights</h4>
                    <p className="text-[11px] text-slate-500">Advanced ML for early risk screening</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-blue-100/60 text-blue-700 flex items-center justify-center flex-shrink-0">
                    <TrendingUp className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-800">Personalized Baselines</h4>
                    <p className="text-[11px] text-slate-500">Track unique health shifts over time</p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-emerald-100/60 text-emerald-700 flex items-center justify-center flex-shrink-0">
                    <Lock className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-800">Secure & Private</h4>
                    <p className="text-[11px] text-slate-500">Strict clinical privacy boundaries</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Visual Card */}
            <div className="lg:col-span-5 relative">
              <div className="relative mx-auto max-w-md bg-white rounded-3xl p-6 shadow-elevated border border-slate-200">
                <div className="flex items-center justify-between pb-4 border-b border-slate-100 mb-6">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-2xl bg-teal-500 text-white flex items-center justify-center font-bold">
                      <Activity className="w-5 h-5" />
                    </div>
                    <div>
                      <h4 className="text-sm font-bold text-slate-800">Clinical Dashboard</h4>
                      <p className="text-xs text-slate-400">Real-time health overview</p>
                    </div>
                  </div>
                  <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                    Active
                  </span>
                </div>

                {/* Metric Previews */}
                <div className="space-y-4">
                  <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100 flex items-center justify-between">
                    <div>
                      <span className="text-xs text-slate-500 font-medium">Blood Glucose</span>
                      <div className="text-xl font-bold text-slate-900 mt-0.5">110 mg/dL</div>
                      <span className="text-[11px] text-teal-600 font-semibold">Stable Baseline: 120.0</span>
                    </div>
                    <div className="text-right">
                      <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                        Normal
                      </span>
                    </div>
                  </div>

                  <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100 flex items-center justify-between">
                    <div>
                      <span className="text-xs text-slate-500 font-medium">Resting Heart Rate</span>
                      <div className="text-xl font-bold text-slate-900 mt-0.5">72 bpm</div>
                      <span className="text-[11px] text-slate-400">Temporal: Morning (08:30)</span>
                    </div>
                    <div className="text-right">
                      <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
                        Resting
                      </span>
                    </div>
                  </div>
                </div>

                {/* Floating Priority Pill */}
                <div className="mt-6 p-4 rounded-2xl bg-gradient-to-r from-teal-600 to-cyan-600 text-white text-xs flex items-center justify-between shadow-sm">
                  <div className="flex items-center gap-2.5">
                    <HeartHandshake className="w-5 h-5 flex-shrink-0" />
                    <span className="font-semibold">Your Health, Our Priority</span>
                  </div>
                  <span className="bg-white/20 px-2 py-0.5 rounded text-[10px] font-bold">24/7 Monitoring</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 2. WHY CHOOSE SMARTHEALTH (Design Reference Section #4) */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto space-y-4 mb-16">
          <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight sm:text-4xl">
            Why Choose <span className="text-teal-600">SmartHealth</span>?
          </h2>
          <p className="text-base text-slate-600 leading-relaxed">
            We combine the power of Artificial Intelligence with personalized health baselines to help you live a healthier, longer, and more informed life.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="bg-white rounded-2xl p-8 border border-slate-200/80 shadow-card hover:shadow-card-hover transition-all space-y-3">
            <div className="w-12 h-12 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center font-bold">
              <Brain className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-800">AI Risk Assessment</h3>
            <p className="text-sm text-slate-500 leading-relaxed">
              Early detection screening for multi-disease conditions based on ML feature screening evaluations.
            </p>
          </div>

          <div className="bg-white rounded-2xl p-8 border border-slate-200/80 shadow-card hover:shadow-card-hover transition-all space-y-3">
            <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center font-bold">
              <TrendingUp className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-800">Personalized Insights</h3>
            <p className="text-sm text-slate-500 leading-relaxed">
              Tailored to your individual biometric profile. No generic one-size-fits-all medical thresholds.
            </p>
          </div>

          <div className="bg-white rounded-2xl p-8 border border-slate-200/80 shadow-card hover:shadow-card-hover transition-all space-y-3">
            <div className="w-12 h-12 rounded-xl bg-cyan-50 text-cyan-600 flex items-center justify-center font-bold">
              <Activity className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-800">Continuous Monitoring</h3>
            <p className="text-sm text-slate-500 leading-relaxed">
              Track longitudinal patterns across days, weeks, and months with immutable snapshot integrity.
            </p>
          </div>

          <div className="bg-white rounded-2xl p-8 border border-slate-200/80 shadow-card hover:shadow-card-hover transition-all space-y-3">
            <div className="w-12 h-12 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center font-bold">
              <MessageSquare className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-800">24/7 AI Clinical Assistant</h3>
            <p className="text-sm text-slate-500 leading-relaxed">
              Structured clinical entity extraction, safe red-flag symptom triage, and medical education answers.
            </p>
          </div>

          <div className="bg-white rounded-2xl p-8 border border-slate-200/80 shadow-card hover:shadow-card-hover transition-all space-y-3">
            <div className="w-12 h-12 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center font-bold">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-800">Secure & Private</h3>
            <p className="text-sm text-slate-500 leading-relaxed">
              End-to-end data isolation, transparent machine learning parameters, and patient-first governance.
            </p>
          </div>

          <div className="bg-white rounded-2xl p-8 border border-slate-200/80 shadow-card hover:shadow-card-hover transition-all space-y-3">
            <div className="w-12 h-12 rounded-xl bg-purple-50 text-purple-600 flex items-center justify-center font-bold">
              <Zap className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-800">Easy to Use</h3>
            <p className="text-sm text-slate-500 leading-relaxed">
              Intuitive responsive patient dashboard designed for seamless logging across desktop and mobile devices.
            </p>
          </div>
        </div>
      </section>

      {/* 3. HOW IT WORKS */}
      <section className="bg-white py-16 border-y border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">
              How the Platform Works
            </h2>
            <p className="text-sm text-slate-500 mt-2">
              A principled, four-stage longitudinal screening and monitoring workflow.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="relative p-6 rounded-2xl bg-slate-50 border border-slate-200/80">
              <span className="text-3xl font-black text-teal-600/30 mb-2 block">01</span>
              <h4 className="text-base font-bold text-slate-800 mb-1">Record Observations</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Log biometric measurements including glucose, heart rate, blood pressure, and activity levels.
              </p>
            </div>

            <div className="relative p-6 rounded-2xl bg-slate-50 border border-slate-200/80">
              <span className="text-3xl font-black text-teal-600/30 mb-2 block">02</span>
              <h4 className="text-base font-bold text-slate-800 mb-1">Build Personal Baseline</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                The platform computes your individual running mean and variability from historical data.
              </p>
            </div>

            <div className="relative p-6 rounded-2xl bg-slate-50 border border-slate-200/80">
              <span className="text-3xl font-black text-teal-600/30 mb-2 block">03</span>
              <h4 className="text-base font-bold text-slate-800 mb-1">Context-Aware Trends</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Analyze persistence, direction, deviations, and temporal patterns with immutable snapshot integrity.
              </p>
            </div>

            <div className="relative p-6 rounded-2xl bg-slate-50 border border-slate-200/80">
              <span className="text-3xl font-black text-teal-600/30 mb-2 block">04</span>
              <h4 className="text-base font-bold text-slate-800 mb-1">Interactive Assistant</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                Review automated safety alerts, consult the conversational assistant, and share structured records.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 4. CLINICAL DISCLAIMER & CTA BANNER */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="rounded-3xl bg-navy-900 text-white p-8 sm:p-12 relative overflow-hidden shadow-elevated">
          <div className="relative z-10 max-w-2xl space-y-4">
            <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
              Ready to take control of your health journey?
            </h2>
            <p className="text-slate-300 text-sm leading-relaxed">
              Join SmartHealth Platform today to start building your personalized health record and longitudinal trend analysis.
            </p>
            <div className="flex flex-wrap gap-3 pt-2">
              <Link to="/register">
                <Button variant="teal" size="md">
                  Register Patient Account
                </Button>
              </Link>
              <Link to="/dashboard">
                <Button variant="outline" size="md" className="border-slate-600 text-white hover:bg-navy-800">
                  Access Portal
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
