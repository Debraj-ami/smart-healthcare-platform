import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { PatientProvider } from './context/PatientContext';

// Layouts
import PublicLayout from './layouts/PublicLayout';
import PatientLayout from './layouts/PatientLayout';

// Public Pages
import LandingPage from './pages/public/LandingPage';
import AboutPage from './pages/public/AboutPage';
import FeaturesPage from './pages/public/FeaturesPage';

// Auth Pages
import LoginPage from './pages/auth/LoginPage';
import RegisterPage from './pages/auth/RegisterPage';

// Patient Portal Pages
import DashboardPage from './pages/patient/DashboardPage';
import ProfilePage from './pages/patient/ProfilePage';
import HealthDataPage from './pages/patient/HealthDataPage';
import AssessmentsPage from './pages/patient/AssessmentsPage';
import MonitoringPage from './pages/patient/MonitoringPage';
import ChatPage from './pages/patient/ChatPage';
import AlertsPage from './pages/patient/AlertsPage';
import HistoryPage from './pages/patient/HistoryPage';

// 404
import NotFoundPage from './pages/NotFoundPage';

export const App: React.FC = () => {
  return (
    <PatientProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Routes with Navbar & Footer */}
          <Route element={<PublicLayout />}>
            <Route path="/" element={<LandingPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/features" element={<FeaturesPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
          </Route>

          {/* Patient Portal Routes with Dark Navy Sidebar & Header */}
          <Route element={<PatientLayout />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route path="/health" element={<HealthDataPage />} />
            <Route path="/assessments" element={<AssessmentsPage />} />
            <Route path="/monitoring" element={<MonitoringPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/alerts" element={<AlertsPage />} />
            <Route path="/history" element={<HistoryPage />} />
          </Route>

          {/* 404 Catch-All */}
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </BrowserRouter>
    </PatientProvider>
  );
};

export default App;
