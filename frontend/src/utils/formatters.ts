export function formatMetricName(metricType: string): string {
  switch (metricType) {
    case 'GLUCOSE':
      return 'Blood Glucose';
    case 'HEART_RATE':
      return 'Heart Rate';
    case 'BLOOD_PRESSURE_SYSTOLIC':
      return 'Systolic Blood Pressure';
    case 'BLOOD_PRESSURE_DIASTOLIC':
      return 'Diastolic Blood Pressure';
    case 'TEMPERATURE':
      return 'Body Temperature';
    case 'OXYGEN_SATURATION':
      return 'Oxygen Saturation (SpO2)';
    case 'WEIGHT':
      return 'Weight';
    case 'HEIGHT':
      return 'Height';
    default:
      return metricType
        .split('_')
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
        .join(' ');
  }
}

export function getDefaultUnit(metricType: string): string {
  switch (metricType) {
    case 'GLUCOSE':
      return 'mg/dL';
    case 'HEART_RATE':
      return 'bpm';
    case 'BLOOD_PRESSURE_SYSTOLIC':
    case 'BLOOD_PRESSURE_DIASTOLIC':
      return 'mmHg';
    case 'TEMPERATURE':
      return '°F';
    case 'OXYGEN_SATURATION':
      return '%';
    case 'WEIGHT':
      return 'kg';
    case 'HEIGHT':
      return 'cm';
    default:
      return '';
  }
}

export function formatDate(dateStr?: string | null): string {
  if (!dateStr) return 'N/A';
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    return d.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return dateStr;
  }
}

export function formatDateTime(dateStr?: string | null): string {
  if (!dateStr) return 'N/A';
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    return d.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return dateStr;
  }
}

export function getTrendDirectionBadge(direction: string): {
  bg: string;
  text: string;
  border: string;
  label: string;
} {
  switch (direction) {
    case 'INCREASING':
      return {
        bg: 'bg-amber-50',
        text: 'text-amber-700',
        border: 'border-amber-200',
        label: 'Increasing',
      };
    case 'DECREASING':
      return {
        bg: 'bg-blue-50',
        text: 'text-blue-700',
        border: 'border-blue-200',
        label: 'Decreasing',
      };
    case 'STABLE':
    default:
      return {
        bg: 'bg-emerald-50',
        text: 'text-emerald-700',
        border: 'border-emerald-200',
        label: 'Stable',
      };
  }
}

export function getAlertSeverityBadge(severity: string): {
  bg: string;
  text: string;
  border: string;
} {
  switch (severity) {
    case 'CRITICAL':
      return { bg: 'bg-rose-50', text: 'text-rose-700', border: 'border-rose-200' };
    case 'HIGH':
      return { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200' };
    case 'MEDIUM':
      return { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200' };
    case 'LOW':
    default:
      return { bg: 'bg-slate-50', text: 'text-slate-700', border: 'border-slate-200' };
  }
}
