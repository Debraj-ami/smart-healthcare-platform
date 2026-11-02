import React from 'react';
import { NavLink, Link } from 'react-router-dom';
import {
  LayoutDashboard,
  User,
  Activity,
  FileCheck2,
  TrendingUp,
  MessageSquare,
  Bell,
  History,
  Settings,
  LogOut,
  HeartPulse,
} from 'lucide-react';

interface PatientSidebarProps {
  onCloseMobile?: () => void;
}

export const PatientSidebar: React.FC<PatientSidebarProps> = ({ onCloseMobile }) => {
  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'Profile', path: '/profile', icon: User },
    { name: 'Health Data', path: '/health', icon: Activity },
    { name: 'Assessments', path: '/assessments', icon: FileCheck2 },
    { name: 'Monitoring', path: '/monitoring', icon: TrendingUp },
    { name: 'AI Assistant', path: '/chat', icon: MessageSquare },
    { name: 'Alerts', path: '/alerts', icon: Bell },
    { name: 'History', path: '/history', icon: History },
  ];

  return (
    <aside className="w-64 bg-navy-900 text-slate-300 flex flex-col h-full border-r border-navy-800">
      {/* Brand Header */}
      <div className="h-18 px-6 flex items-center gap-3 border-b border-navy-800/80">
        <div className="w-10 h-10 rounded-xl bg-teal-500/20 border border-teal-500/30 flex items-center justify-center text-teal-400">
          <HeartPulse className="w-6 h-6" />
        </div>
        <div className="flex flex-col">
          <span className="text-lg font-bold tracking-tight text-white flex items-center gap-1">
            Smart<span className="text-teal-400">Health</span>
          </span>
          <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-400">
            Patient Portal
          </span>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto dark-scroll">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={onCloseMobile}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white hover:bg-navy-800/60'
                }`
              }
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              <span>{item.name}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* Bottom Actions */}
      <div className="p-3 border-t border-navy-800/80 space-y-1">
        <button
          onClick={() => alert('Settings: Patient preference customization available in Phase 12.')}
          className="w-full flex items-center gap-3 px-3.5 py-2 rounded-xl text-sm font-medium text-slate-400 hover:text-white hover:bg-navy-800/60 transition-colors"
        >
          <Settings className="w-5 h-5 flex-shrink-0" />
          <span>Settings</span>
        </button>
        <Link
          to="/login"
          onClick={onCloseMobile}
          className="w-full flex items-center gap-3 px-3.5 py-2 rounded-xl text-sm font-medium text-rose-400/90 hover:text-rose-300 hover:bg-navy-800/60 transition-colors"
        >
          <LogOut className="w-5 h-5 flex-shrink-0" />
          <span>Logout</span>
        </Link>
      </div>
    </aside>
  );
};

export default PatientSidebar;
