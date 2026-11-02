import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Activity, AlertCircle, ArrowRight, Info, ShieldCheck, KeyRound } from 'lucide-react';
import Button from '../../components/common/Button';
import { usePatient } from '../../context/PatientContext';

export const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const { activePatientId, setActivePatientId } = usePatient();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSimulatedSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Do NOT create fake JWT or fake auth tokens
    // Transparently navigate to dashboard with notice
    navigate('/dashboard');
  };

  return (
    <div className="min-h-[calc(100vh-72px)] flex items-center justify-center p-4 sm:p-6 lg:p-8 bg-slate-50">
      <div className="max-w-4xl w-full bg-white rounded-3xl shadow-elevated border border-slate-200 overflow-hidden grid grid-cols-1 lg:grid-cols-12">
        {/* Left Card: Login UI (7 cols) */}
        <div className="lg:col-span-7 p-8 sm:p-10 space-y-6">
          <div>
            <Link to="/" className="inline-flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-xl bg-teal-600 text-white flex items-center justify-center font-bold">
                <Activity className="w-4 h-4" />
              </div>
              <span className="text-lg font-bold text-slate-900">
                Smart<span className="text-teal-600">Health</span>
              </span>
            </Link>
            <h2 className="text-2xl font-extrabold text-slate-900">Welcome Back</h2>
            <p className="text-xs text-slate-500 mt-1">
              Login to access your personalized health dashboard and observation history.
            </p>
          </div>

          {/* Honest Backend Integration Banner */}
          <div className="p-4 rounded-2xl bg-amber-50/80 border border-amber-200 text-amber-900 space-y-2 text-xs leading-relaxed">
            <div className="flex items-center gap-2 font-bold text-amber-800">
              <Info className="w-4 h-4 flex-shrink-0 text-amber-600" />
              <span>Backend Authentication Pending (Spring Boot)</span>
            </div>
            <p>
              The Spring Boot backend currently supports user registration (<code className="font-mono bg-amber-100/70 px-1 py-0.5 rounded">POST /api/users</code>), but full production JWT authentication is not yet integrated.
            </p>
            <p className="text-[11px] text-amber-700">
              You can explore the patient dashboard directly using the verified test patient profiles or create a new user profile via the register page.
            </p>
          </div>

          <form onSubmit={handleSimulatedSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Email Address
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@example.com"
                className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="block text-xs font-semibold text-slate-700">
                  Password
                </label>
                <a
                  href="#forgot"
                  onClick={(e) => {
                    e.preventDefault();
                    alert('Password reset capability scheduled for upcoming release.');
                  }}
                  className="text-xs font-medium text-teal-600 hover:underline"
                >
                  Forgot Password?
                </a>
              </div>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                className="w-full px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
              />
            </div>

            <div className="pt-2 flex flex-col gap-2.5">
              <Button type="submit" variant="teal" size="md" className="w-full">
                Enter Patient Portal (Direct Access)
              </Button>
            </div>

            <div className="text-center text-xs text-slate-500 pt-2">
              Don't have an account?{' '}
              <Link to="/register" className="text-teal-600 font-semibold hover:underline">
                Register here
              </Link>
            </div>
          </form>
        </div>

        {/* Right Info Card (5 cols, matching design reference #2) */}
        <div className="lg:col-span-5 bg-gradient-to-br from-navy-900 via-navy-850 to-teal-900 p-8 sm:p-10 text-white flex flex-col justify-between relative overflow-hidden">
          <div className="relative z-10 space-y-6">
            <span className="text-xs uppercase tracking-wider font-semibold text-teal-300">
              Clinical Quality
            </span>
            <blockquote className="text-xl font-bold leading-snug">
              “A healthier you for a better tomorrow.”
            </blockquote>
            <p className="text-xs text-slate-300 leading-relaxed">
              Track longitudinal observations, establish individualized personal baselines, and analyze context-aware biometric shifts.
            </p>
          </div>

          <div className="relative z-10 pt-8 border-t border-navy-700/80 text-[11px] text-teal-200 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-teal-400" />
            <span>Encrypted HIPAA-Aligned Research Prototype</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
