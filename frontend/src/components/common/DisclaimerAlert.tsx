import React from 'react';
import { AlertTriangle, ShieldCheck } from 'lucide-react';

interface DisclaimerAlertProps {
  variant?: 'warning' | 'info';
  className?: string;
}

export const DisclaimerAlert: React.FC<DisclaimerAlertProps> = ({
  variant = 'info',
  className = '',
}) => {
  return (
    <div
      className={`flex items-start gap-3 p-4 rounded-xl border text-xs leading-relaxed ${
        variant === 'warning'
          ? 'bg-amber-50/70 border-amber-200/80 text-amber-800'
          : 'bg-teal-50/70 border-teal-200/80 text-teal-800'
      } ${className}`}
    >
      {variant === 'warning' ? (
        <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
      ) : (
        <ShieldCheck className="w-4 h-4 text-teal-600 mt-0.5 flex-shrink-0" />
      )}
      <div>
        <span className="font-semibold">Clinical & Research Safety Notice:</span> This platform provides AI-assisted health screening, statistical baseline tracking, and clinical information extraction. It is not an automated medical diagnostic tool and is not a substitute for professional medical evaluation, diagnosis, or treatment.
      </div>
    </div>
  );
};

export default DisclaimerAlert;
