import React, { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import PatientSidebar from '../components/navigation/PatientSidebar';
import PatientHeader from '../components/navigation/PatientHeader';
import DisclaimerAlert from '../components/common/DisclaimerAlert';
import { ChatAssistantProvider } from '../context/ChatAssistantContext';
import FloatingAIAvatar from '../components/chatbot/FloatingAIAvatar';
import FloatingChatDrawer from '../components/chatbot/FloatingChatDrawer';

const PatientLayoutInner: React.FC = () => {
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const location = useLocation();

  // Hide floating avatar when the patient is already on the dedicated /chat page
  const isChatPage = location.pathname === '/chat';

  return (
    <div className="min-h-screen flex bg-[#f8fafc]">
      {/* Desktop Sidebar */}
      <div className="hidden lg:block fixed inset-y-0 left-0 z-40">
        <PatientSidebar />
      </div>

      {/* Mobile Sidebar Drawer with Backdrop */}
      {mobileSidebarOpen && (
        <div className="lg:hidden fixed inset-0 z-50 flex">
          <div
            className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm transition-opacity"
            onClick={() => setMobileSidebarOpen(false)}
          />
          <div className="relative flex-1 flex flex-col max-w-xs w-full bg-navy-900 z-10 shadow-xl">
            <PatientSidebar onCloseMobile={() => setMobileSidebarOpen(false)} />
          </div>
        </div>
      )}

      {/* Main Body */}
      <div className="flex-1 lg:pl-64 flex flex-col min-w-0">
        <PatientHeader onToggleMobileMenu={() => setMobileSidebarOpen(true)} />
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto w-full">
          <Outlet />
          <div className="mt-8">
            <DisclaimerAlert variant="info" />
          </div>
        </main>
      </div>

      {/* Floating Smart Health AI Assistant Drawer */}
      <FloatingChatDrawer />

      {/* Floating Smart Health AI Avatar Launcher */}
      {!isChatPage && <FloatingAIAvatar position="fixed" />}
    </div>
  );
};

export const PatientLayout: React.FC = () => {
  return (
    <ChatAssistantProvider>
      <PatientLayoutInner />
    </ChatAssistantProvider>
  );
};

export default PatientLayout;
