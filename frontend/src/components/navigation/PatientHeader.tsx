import React, { useState } from 'react';
import { Search, Bell, Menu, Users, Activity, CheckCircle2 } from 'lucide-react';
import { usePatient } from '../../context/PatientContext';
import { Link } from 'react-router-dom';

interface PatientHeaderProps {
  onToggleMobileMenu: () => void;
}

export const PatientHeader: React.FC<PatientHeaderProps> = ({ onToggleMobileMenu }) => {
  const { activePatientId, setActivePatientId, currentUser, isLoading } = usePatient();
  const [patientInput, setPatientInput] = useState(activePatientId.toString());

  const handlePatientSwitch = (e: React.FormEvent) => {
    e.preventDefault();
    const id = parseInt(patientInput, 10);
    if (!isNaN(id) && id > 0) {
      setActivePatientId(id);
    }
  };

  const displayName = currentUser
    ? `${currentUser.firstName} ${currentUser.lastName}`
    : isLoading
    ? 'Loading...'
    : `Patient #${activePatientId}`;

  const initials = currentUser
    ? `${currentUser.firstName[0]}${currentUser.lastName[0]}`.toUpperCase()
    : `P${activePatientId}`;

  return (
    <header className="h-18 bg-white border-b border-slate-200 px-4 sm:px-6 lg:px-8 flex items-center justify-between gap-4 sticky top-0 z-30">
      {/* Left: Mobile hamburger & Search bar */}
      <div className="flex items-center gap-3 flex-1 max-w-md">
        <button
          onClick={onToggleMobileMenu}
          className="lg:hidden p-2 rounded-xl text-slate-600 hover:bg-slate-100"
          aria-label="Open sidebar"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="relative w-full">
          <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search health records, metrics, alerts..."
            className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 transition-all"
          />
        </div>
      </div>

      {/* Right: Backend indicator, Patient Switcher, Notifications, Profile */}
      <div className="flex items-center gap-3 sm:gap-4">
        {/* Real Backend Connection Badge */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-[11px] font-medium text-emerald-700">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span>Live Backend (8080)</span>
        </div>

        {/* Patient Switcher */}
        <form onSubmit={handlePatientSwitch} className="flex items-center gap-1.5">
          <label htmlFor="patient-select" className="text-xs text-slate-500 hidden sm:inline flex items-center gap-1">
            <Users className="w-3.5 h-3.5" /> ID:
          </label>
          <select
            id="patient-select"
            value={activePatientId}
            onChange={(e) => {
              const val = parseInt(e.target.value, 10);
              setActivePatientId(val);
              setPatientInput(val.toString());
            }}
            className="text-xs font-semibold py-1.5 px-2.5 bg-slate-50 border border-slate-200 rounded-lg text-slate-700 focus:outline-none focus:border-teal-500 cursor-pointer"
          >
            <option value={4}>Patient #4 (Glucose Trend Demo)</option>
            <option value={2}>Patient #2 (Historical Context)</option>
            <option value={3}>Patient #3 (Recent Audit)</option>
            <option value={1}>Patient #1</option>
          </select>
        </form>

        {/* Notification Bell */}
        <Link
          to="/alerts"
          className="p-2 rounded-xl text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors relative"
          title="View active alerts"
        >
          <Bell className="w-5 h-5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-rose-500 rounded-full"></span>
        </Link>

        {/* User Profile Pill */}
        <Link
          to="/profile"
          className="flex items-center gap-2.5 pl-2 py-1 pr-1 sm:pr-3 rounded-full hover:bg-slate-50 transition-colors border border-transparent hover:border-slate-200"
        >
          <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-blue-600 to-teal-500 text-white flex items-center justify-center font-bold text-xs shadow-sm">
            {initials}
          </div>
          <div className="hidden sm:flex flex-col text-left">
            <span className="text-xs font-semibold text-slate-800 leading-tight">
              {displayName}
            </span>
            <span className="text-[10px] text-slate-400 capitalize">
              {currentUser?.role?.toLowerCase() || 'Patient'}
            </span>
          </div>
        </Link>
      </div>
    </header>
  );
};

export default PatientHeader;
