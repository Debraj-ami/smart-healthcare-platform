import React from 'react';
import { Link } from 'react-router-dom';
import { FileQuestion, ArrowLeft, Home } from 'lucide-react';
import Button from '../components/common/Button';

export const NotFoundPage: React.FC = () => {
  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center p-6 text-center">
      <div className="w-16 h-16 rounded-2xl bg-teal-50 text-teal-600 flex items-center justify-center mb-4">
        <FileQuestion className="w-8 h-8" />
      </div>
      <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight mb-2">404 — Page Not Found</h1>
      <p className="text-sm text-slate-500 max-w-sm mb-6 leading-relaxed">
        The clinical route or page you requested does not exist or has been moved.
      </p>
      <div className="flex gap-3">
        <Link to="/">
          <Button variant="outline" size="sm" leftIcon={<Home className="w-4 h-4" />}>
            Home
          </Button>
        </Link>
        <Link to="/dashboard">
          <Button variant="teal" size="sm">
            Patient Dashboard
          </Button>
        </Link>
      </div>
    </div>
  );
};

export default NotFoundPage;
