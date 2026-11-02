import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  X,
  Heart,
  AlertCircle,
  ArrowLeft,
  Info,
  TrendingUp,
  TrendingDown,
  Activity,
  History,
  HelpCircle,
  FileText,
} from 'lucide-react';
import { HeartDiseaseFeatures } from '../../types/assessment';
import Button from '../common/Button';
import { usePatient } from '../../context/PatientContext';
import heartPredictionService, {
  HeartPredictResponse,
} from '../../services/heartPredictionService';
import { formatDateTime } from '../../utils/formatters';

interface HeartDiseaseAssessmentModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAssessmentComplete?: () => void;
  defaultAge?: number | null;
  defaultGender?: string | null;
}

const INITIAL_FORM: HeartDiseaseFeatures = {
  age: '',
  sex: '',
  cp: '',
  trestbps: '',
  chol: '',
  fbs: '',
  restecg: '',
  thalach: '',
  exang: '',
  oldpeak: '',
  slope: '',
  ca: '',
  thal: '',
};

type FormErrors = Partial<Record<keyof HeartDiseaseFeatures, string>>;
type ModalView = 'form' | 'review' | 'loading' | 'result' | 'error';

// -----------------------------------------------------------------------
// Helper: Map categorical codes to human-readable labels
// -----------------------------------------------------------------------
const LABELS = {
  sex: {
    '0': 'Female',
    '1': 'Male',
  },
  cp: {
    '1': 'Typical angina',
    '2': 'Atypical angina',
    '3': 'Non-anginal pain',
    '4': 'Asymptomatic',
  },
  fbs: {
    '0': '<= 120 mg/dL',
    '1': '> 120 mg/dL',
  },
  restecg: {
    '0': 'Normal',
    '1': 'ST-T wave abnormality',
    '2': 'Left ventricular hypertrophy',
  },
  exang: {
    '0': 'No',
    '1': 'Yes',
  },
  slope: {
    '1': 'Upsloping',
    '2': 'Flat',
    '3': 'Downsloping',
  },
  ca: {
    '': 'Not provided (null)',
    'null': 'Not provided (null)',
    '0': '0 vessels',
    '1': '1 vessel',
    '2': '2 vessels',
    '3': '3 vessels',
  },
  thal: {
    '': 'Not provided (null)',
    'null': 'Not provided (null)',
    '3': 'Normal (3.0)',
    '6': 'Fixed defect (6.0)',
    '7': 'Reversible defect (7.0)',
  },
} as const;

// -----------------------------------------------------------------------
// Component: SHAP contribution bar
// -----------------------------------------------------------------------
function ShapBar({ contribution, maxAbsContrib }: { contribution: number; maxAbsContrib: number }) {
  const pct = maxAbsContrib > 0 ? Math.min(100, Math.round((Math.abs(contribution) / maxAbsContrib) * 100)) : 0;
  const isPositive = contribution > 0;
  const color = isPositive ? 'bg-rose-500' : contribution < 0 ? 'bg-emerald-500' : 'bg-slate-300';

  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono font-semibold text-slate-600 w-16 text-right">
        {isPositive ? '+' : ''}{contribution.toFixed(4)}
      </span>
    </div>
  );
}

// -----------------------------------------------------------------------
// Main Heart Assessment Modal
// -----------------------------------------------------------------------
export const HeartDiseaseAssessmentModal: React.FC<HeartDiseaseAssessmentModalProps> = ({
  isOpen,
  onClose,
  onAssessmentComplete,
  defaultAge,
  defaultGender,
}) => {
  const { activePatientId, currentUser } = usePatient();
  const navigate = useNavigate();
  const [formData, setFormData] = useState<HeartDiseaseFeatures>(INITIAL_FORM);
  const [errors, setErrors] = useState<FormErrors>({});
  const [view, setView] = useState<ModalView>('form');
  const [result, setResult] = useState<HeartPredictResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  useEffect(() => {
    if (isOpen) {
      // Determine initial sex from defaultGender if available
      let initialSex = '';
      if (defaultGender) {
        const g = defaultGender.trim().toUpperCase();
        if (g === 'MALE' || g === 'M') initialSex = '1';
        else if (g === 'FEMALE' || g === 'F') initialSex = '0';
      }

      setFormData({
        ...INITIAL_FORM,
        age: defaultAge ? String(defaultAge) : '',
        sex: initialSex,
      });
      setErrors({});
      setView('form');
      setResult(null);
      setErrorMessage('');
      setIsSubmitting(false);
    }
  }, [isOpen, defaultAge, defaultGender]);

  if (!isOpen) return null;

  const handleChange = (field: keyof HeartDiseaseFeatures, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: undefined }));
    }
  };

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    // 1. Age (1 - 120)
    const ageNum = Number(formData.age);
    if (!formData.age.trim()) {
      newErrors.age = 'Age is required (1–120 years)';
    } else if (isNaN(ageNum) || !Number.isInteger(ageNum) || ageNum < 1 || ageNum > 120) {
      newErrors.age = 'Age must be an integer between 1 and 120';
    }

    // 2. Sex (0, 1)
    if (!formData.sex) {
      newErrors.sex = 'Please select biological sex';
    } else if (formData.sex !== '0' && formData.sex !== '1') {
      newErrors.sex = 'Sex must be Female (0) or Male (1)';
    }

    // 3. Chest pain type (1, 2, 3, 4)
    if (!formData.cp) {
      newErrors.cp = 'Please select chest pain type';
    } else if (!['1', '2', '3', '4'].includes(formData.cp)) {
      newErrors.cp = 'Invalid chest pain selection';
    }

    // 4. Resting blood pressure (50 - 300)
    const bpNum = Number(formData.trestbps);
    if (!formData.trestbps.trim()) {
      newErrors.trestbps = 'Resting blood pressure is required';
    } else if (isNaN(bpNum) || bpNum < 50 || bpNum > 300) {
      newErrors.trestbps = 'Must be between 50 and 300 mm Hg';
    }

    // 5. Serum cholesterol (50 - 800)
    const cholNum = Number(formData.chol);
    if (!formData.chol.trim()) {
      newErrors.chol = 'Serum cholesterol is required';
    } else if (isNaN(cholNum) || cholNum < 50 || cholNum > 800) {
      newErrors.chol = 'Must be between 50 and 800 mg/dL';
    }

    // 6. Fasting blood sugar (0, 1)
    if (formData.fbs === '') {
      newErrors.fbs = 'Please select fasting blood sugar range';
    } else if (formData.fbs !== '0' && formData.fbs !== '1') {
      newErrors.fbs = 'Invalid fasting blood sugar selection';
    }

    // 7. Resting ECG (0, 1, 2)
    if (formData.restecg === '') {
      newErrors.restecg = 'Please select resting ECG result';
    } else if (!['0', '1', '2'].includes(formData.restecg)) {
      newErrors.restecg = 'Invalid resting ECG selection';
    }

    // 8. Maximum heart rate achieved (50 - 250)
    const hrNum = Number(formData.thalach);
    if (!formData.thalach.trim()) {
      newErrors.thalach = 'Maximum heart rate is required';
    } else if (isNaN(hrNum) || hrNum < 50 || hrNum > 250) {
      newErrors.thalach = 'Must be between 50 and 250 bpm';
    }

    // 9. Exercise-induced angina (0, 1)
    if (formData.exang === '') {
      newErrors.exang = 'Please select exercise angina status';
    } else if (formData.exang !== '0' && formData.exang !== '1') {
      newErrors.exang = 'Invalid exercise angina selection';
    }

    // 10. ST depression oldpeak (0.0 - 15.0)
    const opNum = Number(formData.oldpeak);
    if (!formData.oldpeak.trim()) {
      newErrors.oldpeak = 'ST depression value is required';
    } else if (isNaN(opNum) || opNum < 0 || opNum > 15) {
      newErrors.oldpeak = 'Must be between 0.0 and 15.0 mm';
    }

    // 11. Slope of peak ST segment (1, 2, 3)
    if (!formData.slope) {
      newErrors.slope = 'Please select ST segment slope';
    } else if (!['1', '2', '3'].includes(formData.slope)) {
      newErrors.slope = 'Invalid slope selection';
    }

    // 12. CA (Major vessels colored, nullable: '', 'null', '0', '1', '2', '3')
    if (formData.ca && formData.ca !== 'null' && !['0', '1', '2', '3'].includes(formData.ca)) {
      newErrors.ca = 'Must be 0, 1, 2, 3 vessels, or Not provided';
    }

    // 13. Thal (Thallium scintigraphy, nullable: '', 'null', '3', '6', '7')
    if (formData.thal && formData.thal !== 'null' && !['3', '6', '7'].includes(formData.thal)) {
      newErrors.thal = 'Must be Normal (3), Fixed (6), Reversible (7), or Not provided';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleProceedToReview = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      setView('review');
    }
  };

  const handleConfirmAndRun = async () => {
    if (isSubmitting) return; // Prevent duplicate submission
    setIsSubmitting(true);
    setView('loading');

    try {
      // Parse values according to contract; nullable ca and thal produce null if unselected
      const payloadCa = formData.ca && formData.ca !== 'null' ? parseFloat(formData.ca) : null;
      const payloadThal = formData.thal && formData.thal !== 'null' ? parseFloat(formData.thal) : null;

      const prediction = await heartPredictionService.predict({
        patientId: activePatientId && activePatientId > 0 ? activePatientId : undefined,
        age: parseFloat(formData.age),
        sex: parseFloat(formData.sex),
        cp: parseFloat(formData.cp),
        trestbps: parseFloat(formData.trestbps),
        chol: parseFloat(formData.chol),
        fbs: parseFloat(formData.fbs),
        restecg: parseFloat(formData.restecg),
        thalach: parseFloat(formData.thalach),
        exang: parseFloat(formData.exang),
        oldpeak: parseFloat(formData.oldpeak),
        slope: parseFloat(formData.slope),
        ca: payloadCa,
        thal: payloadThal,
      });

      setResult(prediction);
      setView('result');

      if (onAssessmentComplete) {
        onAssessmentComplete();
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'An unexpected error occurred during prediction.';
      setErrorMessage(msg);
      setView('error');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRetry = () => {
    setView('form');
    setErrors({});
    setErrorMessage('');
    setIsSubmitting(false);
  };

  // -----------------------------------------------------------------------
  // Render: 13-Input Form View
  // -----------------------------------------------------------------------
  const renderForm = () => (
    <form id="heart-assessment-form" onSubmit={handleProceedToReview} className="space-y-6">
      {/* Patient Association Notice */}
      <div className="p-3 bg-slate-50 border border-slate-200 rounded-xl text-xs flex items-center justify-between">
        <div className="flex items-center gap-2 text-slate-700">
          <History className="w-4 h-4 text-rose-500" />
          <span>
            Assessing Patient: <strong>#{activePatientId}</strong>{' '}
            {currentUser?.firstName ? `(${currentUser.firstName} ${currentUser.lastName})` : ''}
          </span>
        </div>
        <span className="text-[11px] text-slate-400">Context & History Linked</span>
      </div>

      {/* Clinical Data Screening Guidance & Missing Values Card */}
      <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-700 space-y-2.5">
        <div className="flex items-center gap-2 font-bold text-slate-900">
          <Info className="w-4 h-4 text-rose-500 flex-shrink-0" />
          <span>Clinical Data Screening Assessment</span>
        </div>
        <p className="leading-relaxed text-slate-600">
          This model uses structured clinical and laboratory information. If you have a recent medical report or validated measurements, enter those values here. Do not guess unknown values.
        </p>

        <div className="pt-2 border-t border-slate-200/80">
          <p className="font-semibold text-slate-800 mb-1">Don't have all of these values?</p>
          <p className="text-slate-500 leading-relaxed mb-2.5">
            That's okay. You can still use Smart Health for symptom conversations, health monitoring, personal baselines, trends, and previous assessments. The Heart screening model can be used when the required clinical information is available.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => {
                onClose();
                navigate('/chat');
              }}
              className="px-3 py-1 rounded-lg bg-white border border-slate-200 text-teal-700 font-semibold text-[11px] hover:bg-slate-50 transition-colors shadow-2xs"
            >
              Talk about symptoms →
            </button>
            <button
              type="button"
              onClick={() => {
                onClose();
                navigate('/monitoring');
              }}
              className="px-3 py-1 rounded-lg bg-white border border-slate-200 text-blue-700 font-semibold text-[11px] hover:bg-slate-50 transition-colors shadow-2xs"
            >
              Monitor my health →
            </button>
          </div>
        </div>
      </div>

      {/* SECTION 1: Demographic & Vital Signs */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 pb-1 border-b border-slate-100">
          <span className="w-2 h-2 rounded-full bg-rose-500" />
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            1. Demographics & Vital Signs
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {/* Age */}
          <div>
            <label htmlFor="heart-age" className="block text-xs font-semibold text-slate-700 mb-1">
              Age <span className="text-rose-500">*</span>
            </label>
            <div className="relative">
              <input
                id="heart-age"
                type="number"
                min="1"
                max="120"
                step="1"
                value={formData.age}
                onChange={(e) => handleChange('age', e.target.value)}
                placeholder="e.g. 58"
                className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                  errors.age
                    ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                    : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
                }`}
              />
              <span className="absolute right-3 top-2.5 text-xs text-slate-400">years</span>
            </div>
            {errors.age && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.age}
              </p>
            )}
          </div>

          {/* Sex */}
          <div>
            <label htmlFor="heart-sex" className="block text-xs font-semibold text-slate-700 mb-1">
              Biological Sex <span className="text-rose-500">*</span>
            </label>
            <select
              id="heart-sex"
              value={formData.sex}
              onChange={(e) => handleChange('sex', e.target.value)}
              className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                errors.sex
                  ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                  : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
              }`}
            >
              <option value="">Select Sex</option>
              <option value="1">Male (1)</option>
              <option value="0">Female (0)</option>
            </select>
            {errors.sex && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.sex}
              </p>
            )}
          </div>

          {/* Resting BP (trestbps) */}
          <div>
            <label htmlFor="heart-trestbps" className="block text-xs font-semibold text-slate-700 mb-1">
              Resting BP (trestbps) <span className="text-rose-500">*</span>
            </label>
            <div className="relative">
              <input
                id="heart-trestbps"
                type="number"
                min="50"
                max="300"
                step="1"
                value={formData.trestbps}
                onChange={(e) => handleChange('trestbps', e.target.value)}
                placeholder="e.g. 130"
                className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                  errors.trestbps
                    ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                    : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
                }`}
              />
              <span className="absolute right-3 top-2.5 text-xs text-slate-400">mm Hg</span>
            </div>
            {errors.trestbps && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.trestbps}
              </p>
            )}
          </div>

          {/* Serum Cholesterol (chol) */}
          <div>
            <label htmlFor="heart-chol" className="block text-xs font-semibold text-slate-700 mb-1">
              Serum Cholesterol (chol) <span className="text-rose-500">*</span>
            </label>
            <div className="relative">
              <input
                id="heart-chol"
                type="number"
                min="50"
                max="800"
                step="1"
                value={formData.chol}
                onChange={(e) => handleChange('chol', e.target.value)}
                placeholder="e.g. 240"
                className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                  errors.chol
                    ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                    : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
                }`}
              />
              <span className="absolute right-3 top-2.5 text-xs text-slate-400">mg/dL</span>
            </div>
            {errors.chol && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.chol}
              </p>
            )}
          </div>

          {/* Max Heart Rate Achieved (thalach) */}
          <div>
            <label htmlFor="heart-thalach" className="block text-xs font-semibold text-slate-700 mb-1">
              Max Heart Rate (thalach) <span className="text-rose-500">*</span>
            </label>
            <div className="relative">
              <input
                id="heart-thalach"
                type="number"
                min="50"
                max="250"
                step="1"
                value={formData.thalach}
                onChange={(e) => handleChange('thalach', e.target.value)}
                placeholder="e.g. 150"
                className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                  errors.thalach
                    ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                    : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
                }`}
              />
              <span className="absolute right-3 top-2.5 text-xs text-slate-400">bpm</span>
            </div>
            {errors.thalach && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.thalach}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* SECTION 2: Symptoms & Baseline Diagnostics */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 pb-1 border-b border-slate-100">
          <span className="w-2 h-2 rounded-full bg-rose-500" />
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            2. Symptoms & Diagnostic Findings
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {/* Chest Pain Type (cp) */}
          <div>
            <label htmlFor="heart-cp" className="block text-xs font-semibold text-slate-700 mb-1">
              Chest Pain Type (cp) <span className="text-rose-500">*</span>
            </label>
            <select
              id="heart-cp"
              value={formData.cp}
              onChange={(e) => handleChange('cp', e.target.value)}
              className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                errors.cp
                  ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                  : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
              }`}
            >
              <option value="">Select Chest Pain</option>
              <option value="1">1: Typical angina</option>
              <option value="2">2: Atypical angina</option>
              <option value="3">3: Non-anginal pain</option>
              <option value="4">4: Asymptomatic</option>
            </select>
            {errors.cp && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.cp}
              </p>
            )}
          </div>

          {/* Fasting Blood Sugar (fbs) */}
          <div>
            <label htmlFor="heart-fbs" className="block text-xs font-semibold text-slate-700 mb-1">
              Fasting Blood Sugar (fbs) <span className="text-rose-500">*</span>
            </label>
            <select
              id="heart-fbs"
              value={formData.fbs}
              onChange={(e) => handleChange('fbs', e.target.value)}
              className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                errors.fbs
                  ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                  : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
              }`}
            >
              <option value="">Select Fasting Sugar</option>
              <option value="0">&lt;= 120 mg/dL (0)</option>
              <option value="1">&gt; 120 mg/dL (1)</option>
            </select>
            {errors.fbs && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.fbs}
              </p>
            )}
          </div>

          {/* Resting ECG (restecg) */}
          <div>
            <label htmlFor="heart-restecg" className="block text-xs font-semibold text-slate-700 mb-1">
              Resting ECG (restecg) <span className="text-rose-500">*</span>
            </label>
            <select
              id="heart-restecg"
              value={formData.restecg}
              onChange={(e) => handleChange('restecg', e.target.value)}
              className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                errors.restecg
                  ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                  : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
              }`}
            >
              <option value="">Select Resting ECG</option>
              <option value="0">0: Normal</option>
              <option value="1">1: ST-T wave abnormality</option>
              <option value="2">2: Left ventricular hypertrophy</option>
            </select>
            {errors.restecg && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.restecg}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* SECTION 3: Exercise Stress Test & Fluoroscopy */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 pb-1 border-b border-slate-100">
          <span className="w-2 h-2 rounded-full bg-rose-500" />
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
            3. Exercise Stress Test & Fluoroscopy
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {/* Exercise Angina (exang) */}
          <div>
            <label htmlFor="heart-exang" className="block text-xs font-semibold text-slate-700 mb-1">
              Exercise Angina (exang) <span className="text-rose-500">*</span>
            </label>
            <select
              id="heart-exang"
              value={formData.exang}
              onChange={(e) => handleChange('exang', e.target.value)}
              className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                errors.exang
                  ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                  : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
              }`}
            >
              <option value="">Select Exercise Angina</option>
              <option value="0">No (0)</option>
              <option value="1">Yes (1)</option>
            </select>
            {errors.exang && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.exang}
              </p>
            )}
          </div>

          {/* ST Depression Oldpeak */}
          <div>
            <label htmlFor="heart-oldpeak" className="block text-xs font-semibold text-slate-700 mb-1">
              ST Depression (oldpeak) <span className="text-rose-500">*</span>
            </label>
            <div className="relative">
              <input
                id="heart-oldpeak"
                type="number"
                min="0.0"
                max="15.0"
                step="0.1"
                value={formData.oldpeak}
                onChange={(e) => handleChange('oldpeak', e.target.value)}
                placeholder="e.g. 1.5"
                className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                  errors.oldpeak
                    ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                    : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
                }`}
              />
              <span className="absolute right-3 top-2.5 text-xs text-slate-400">mm</span>
            </div>
            {errors.oldpeak && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.oldpeak}
              </p>
            )}
          </div>

          {/* Slope of peak exercise ST segment */}
          <div>
            <label htmlFor="heart-slope" className="block text-xs font-semibold text-slate-700 mb-1">
              ST Slope (slope) <span className="text-rose-500">*</span>
            </label>
            <select
              id="heart-slope"
              value={formData.slope}
              onChange={(e) => handleChange('slope', e.target.value)}
              className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                errors.slope
                  ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                  : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
              }`}
            >
              <option value="">Select Slope</option>
              <option value="1">1: Upsloping</option>
              <option value="2">2: Flat</option>
              <option value="3">3: Downsloping</option>
            </select>
            {errors.slope && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.slope}
              </p>
            )}
          </div>

          {/* Major Vessels Colored (ca - Nullable) */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label htmlFor="heart-ca" className="block text-xs font-semibold text-slate-700">
                Major Vessels (ca)
              </label>
              <span className="text-[10px] text-slate-400">Nullable</span>
            </div>
            <select
              id="heart-ca"
              value={formData.ca}
              onChange={(e) => handleChange('ca', e.target.value)}
              className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                errors.ca
                  ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                  : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
              }`}
            >
              <option value="null">Not provided (null)</option>
              <option value="0">0 vessels</option>
              <option value="1">1 vessel</option>
              <option value="2">2 vessels</option>
              <option value="3">3 vessels</option>
            </select>
            {errors.ca && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.ca}
              </p>
            )}
          </div>

          {/* Thallium Scintigraphy (thal - Nullable) */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label htmlFor="heart-thal" className="block text-xs font-semibold text-slate-700">
                Thallium Defect (thal)
              </label>
              <span className="text-[10px] text-slate-400">Nullable</span>
            </div>
            <select
              id="heart-thal"
              value={formData.thal}
              onChange={(e) => handleChange('thal', e.target.value)}
              className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                errors.thal
                  ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                  : 'border-slate-200 focus:ring-rose-500/20 focus:border-rose-500'
              }`}
            >
              <option value="null">Not provided (null)</option>
              <option value="3">Normal (3.0)</option>
              <option value="6">Fixed defect (6.0)</option>
              <option value="7">Reversible defect (7.0)</option>
            </select>
            {errors.thal && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                {errors.thal}
              </p>
            )}
          </div>
        </div>
      </div>
    </form>
  );

  // -----------------------------------------------------------------------
  // Render: Review Screen (Step 13)
  // -----------------------------------------------------------------------
  const renderReview = () => {
    const payloadCa = formData.ca && formData.ca !== 'null' ? parseFloat(formData.ca) : null;
    const payloadThal = formData.thal && formData.thal !== 'null' ? parseFloat(formData.thal) : null;

    const items = [
      { label: '1. Age', value: `${formData.age} years`, code: formData.age },
      { label: '2. Biological Sex', value: LABELS.sex[formData.sex as '0' | '1'] || formData.sex, code: formData.sex },
      { label: '3. Chest Pain Type (cp)', value: LABELS.cp[formData.cp as '1' | '2' | '3' | '4'] || formData.cp, code: formData.cp },
      { label: '4. Resting Blood Pressure', value: `${formData.trestbps} mm Hg`, code: formData.trestbps },
      { label: '5. Serum Cholesterol', value: `${formData.chol} mg/dL`, code: formData.chol },
      { label: '6. Fasting Blood Sugar', value: LABELS.fbs[formData.fbs as '0' | '1'] || formData.fbs, code: formData.fbs },
      { label: '7. Resting ECG', value: LABELS.restecg[formData.restecg as '0' | '1' | '2'] || formData.restecg, code: formData.restecg },
      { label: '8. Max Heart Rate (thalach)', value: `${formData.thalach} bpm`, code: formData.thalach },
      { label: '9. Exercise Angina (exang)', value: LABELS.exang[formData.exang as '0' | '1'] || formData.exang, code: formData.exang },
      { label: '10. ST Depression (oldpeak)', value: `${formData.oldpeak} mm`, code: formData.oldpeak },
      { label: '11. ST Segment Slope', value: LABELS.slope[formData.slope as '1' | '2' | '3'] || formData.slope, code: formData.slope },
      {
        label: '12. Major Vessels (ca)',
        value: payloadCa !== null ? LABELS.ca[String(payloadCa) as '0' | '1' | '2' | '3'] : 'Not provided (null)',
        code: payloadCa !== null ? String(payloadCa) : 'null',
        nullable: true,
      },
      {
        label: '13. Thallium Scintigraphy (thal)',
        value: payloadThal !== null ? LABELS.thal[String(payloadThal) as '3' | '6' | '7'] : 'Not provided (null)',
        code: payloadThal !== null ? String(payloadThal) : 'null',
        nullable: true,
      },
    ];

    return (
      <div className="space-y-5 animate-in fade-in duration-200">
        <div className="p-4 rounded-xl bg-blue-50/70 border border-blue-200 text-blue-950 text-xs space-y-1">
          <div className="flex items-center gap-1.5 font-bold text-blue-800">
            <FileText className="w-4 h-4 text-blue-600 flex-shrink-0" />
            <span>Pre-Inference Clinical Input Verification</span>
          </div>
          <p>
            Please review the exact 13 feature inputs below before running ML inference. The target variable (<code>num</code>) is strictly omitted.
          </p>
        </div>

        {/* Patient association tag */}
        <div className="p-3 bg-slate-50 border border-slate-200 rounded-xl text-xs flex items-center justify-between">
          <span className="text-slate-600">
            Target Patient: <strong>#{activePatientId}</strong>
          </span>
          <span className="text-[11px] font-mono text-slate-500">POST /api/assessments/heart/predict</span>
        </div>

        {/* Table of all 13 values */}
        <div className="border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <table className="w-full text-left text-xs divide-y divide-slate-200">
            <thead className="bg-slate-100/70 font-bold text-slate-700">
              <tr>
                <th className="px-3.5 py-2.5">Predictor Feature</th>
                <th className="px-3.5 py-2.5">Human-Readable Value</th>
                <th className="px-3.5 py-2.5 text-right font-mono">Payload Code</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {items.map((it, idx) => (
                <tr key={idx} className="hover:bg-slate-50/60 transition-colors">
                  <td className="px-3.5 py-2 font-medium text-slate-800">{it.label}</td>
                  <td className="px-3.5 py-2 text-slate-600">
                    {it.nullable && it.code === 'null' ? (
                      <span className="italic text-slate-400">Not provided (null)</span>
                    ) : (
                      it.value
                    )}
                  </td>
                  <td className="px-3.5 py-2 text-right font-mono text-slate-700">
                    {it.code}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // -----------------------------------------------------------------------
  // Render: Loading Screen (Step 8)
  // -----------------------------------------------------------------------
  const renderLoading = () => (
    <div className="py-16 text-center space-y-4 animate-in fade-in">
      <div className="relative w-14 h-14 mx-auto">
        <div className="w-14 h-14 rounded-full border-4 border-rose-200 border-t-rose-600 animate-spin" />
        <Heart className="w-6 h-6 text-rose-500 absolute inset-0 m-auto animate-pulse" />
      </div>
      <div className="space-y-1">
        <h4 className="text-sm font-bold text-slate-900">Executing Heart ML Inference</h4>
        <p className="text-xs text-slate-500 max-w-sm mx-auto">
          Calling Spring Boot orchestration and the Python FastAPI microservice (Logistic Regression, threshold t=0.31, SHAP log-odds attribution)...
        </p>
      </div>
      <p className="text-[11px] text-slate-400 font-mono">Single execution guaranteed · No duplicate submissions</p>
    </div>
  );

  // -----------------------------------------------------------------------
  // Render: Error Screen (Step 12)
  // -----------------------------------------------------------------------
  const renderError = () => (
    <div className="py-8 space-y-4 animate-in fade-in">
      <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-950 space-y-2">
        <div className="flex items-center gap-2 font-bold text-rose-800">
          <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0" />
          <span>Heart Assessment Failed</span>
        </div>
        <p className="text-xs text-rose-800 leading-relaxed">{errorMessage}</p>
      </div>

      <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-2">
        <h5 className="font-bold text-slate-800 flex items-center gap-1.5">
          <HelpCircle className="w-4 h-4 text-slate-500" /> Troubleshooting Guidance:
        </h5>
        <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-600">
          <li>Ensure Spring Boot backend is running on port 8080 (<code>cd backend && .\mvnw.cmd spring-boot:run</code>).</li>
          <li>Ensure Python ML microservice is running on port 8001 (<code>cd ml-service && uvicorn app.main:app --port 8001</code>).</li>
          <li>If the error mentions a 404, verify Patient ID #{activePatientId} exists in the database.</li>
          <li>If the error indicates validation, check that all 11 required fields are within range.</li>
        </ul>
      </div>
    </div>
  );

  // -----------------------------------------------------------------------
  // Render: Result Screen (Steps 9, 10, 11)
  // -----------------------------------------------------------------------
  const renderResult = (res: HeartPredictResponse) => {
    const probPct = (res.probability * 100).toFixed(1);
    const isAboveThreshold = res.prediction === 1;

    // SHAP max contribution for bar scaling
    const allContribs = res.allContributions || [];
    const maxAbsContrib = allContribs.length > 0
      ? Math.max(...allContribs.map((c) => c.absContribution || Math.abs(c.contribution)), 0.0001)
      : 0.0001;

    return (
      <div className="space-y-6 animate-in fade-in duration-200">
        {/* Probability & Screening Decision Card */}
        <div className={`p-5 rounded-2xl border ${
          isAboveThreshold
            ? 'bg-rose-50/80 border-rose-200 text-rose-950'
            : 'bg-emerald-50/80 border-emerald-200 text-emerald-950'
        } space-y-4`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className={`w-9 h-9 rounded-xl flex items-center justify-center font-bold ${
                isAboveThreshold ? 'bg-rose-100 text-rose-700' : 'bg-emerald-100 text-emerald-700'
              }`}>
                <Heart className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-sm font-bold">
                  {isAboveThreshold ? 'Positive Screening Result' : 'Negative Screening Result'}
                </h4>
                <p className="text-xs opacity-80">
                  Decision Threshold: t = {res.threshold.toFixed(2)}
                </p>
              </div>
            </div>

            <span className={`px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
              isAboveThreshold
                ? 'bg-rose-200/80 text-rose-800'
                : 'bg-emerald-200/80 text-emerald-800'
            }`}>
              {isAboveThreshold ? 'Threshold Exceeded' : 'Below Threshold'}
            </span>
          </div>

          {/* Model Probability Bar */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs font-semibold">
              <span>Model-Estimated Probability</span>
              <span className="text-xl font-extrabold">{probPct}%</span>
            </div>
            <div className="h-3 bg-white/70 rounded-full overflow-hidden border border-black/5 relative">
              {/* Threshold indicator line at 31% */}
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-slate-900 z-10"
                style={{ left: `${res.threshold * 100}%` }}
                title={`Threshold t=${res.threshold}`}
              />
              <div
                className={`h-full rounded-full transition-all duration-700 ${
                  isAboveThreshold ? 'bg-rose-500' : 'bg-emerald-500'
                }`}
                style={{ width: `${Math.min(100, Math.max(0, res.probability * 100))}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-[11px] opacity-75">
              <span>Raw Model Probability: <strong className="font-mono">{res.probability.toFixed(4)}</strong></span>
              <span>Screening Threshold: <strong className="font-mono">t = {res.threshold}</strong></span>
            </div>
          </div>

          <p className="text-[11px] leading-relaxed opacity-90 border-t border-black/5 pt-2">
            {res.thresholdDescription}
          </p>
        </div>

        {/* Patient Context & Longitudinal History (Step 11) */}
        {res.patientContext && (
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
            <div className="flex items-center justify-between pb-1 border-b border-slate-200">
              <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
                <History className="w-4 h-4 text-rose-500" />
                <span>Patient Context & Personalization State</span>
              </div>
              <span className="text-[11px] font-mono text-slate-500">
                Patient #{res.patientContext.patientId}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-2.5 bg-white rounded-lg border border-slate-100">
                <span className="text-slate-400 block text-[11px]">Previous Heart Assessments</span>
                <span className="font-bold text-slate-800 text-sm">
                  {res.patientContext.previousAssessmentCount} on record
                </span>
              </div>
              <div className="p-2.5 bg-white rounded-lg border border-slate-100">
                <span className="text-slate-400 block text-[11px]">Last Estimated Probability</span>
                <span className="font-bold text-slate-800 text-sm">
                  {res.patientContext.lastProbability !== null
                    ? `${(res.patientContext.lastProbability * 100).toFixed(1)}%`
                    : 'None recorded (first assessment)'}
                </span>
              </div>
              <div className="p-2.5 bg-white rounded-lg border border-slate-100">
                <span className="text-slate-400 block text-[11px]">Last Assessment Date</span>
                <span className="font-semibold text-slate-700">
                  {res.patientContext.lastAssessedAt
                    ? formatDateTime(res.patientContext.lastAssessedAt)
                    : 'N/A'}
                </span>
              </div>
              <div className="p-2.5 bg-white rounded-lg border border-slate-100">
                <span className="text-slate-400 block text-[11px]">Established Personal Baselines</span>
                <span className="font-semibold text-slate-700">
                  {res.patientContext.hasEstablishedBaselines
                    ? res.patientContext.establishedBaselineMetrics.join(', ') || 'Active'
                    : 'None established'}
                </span>
              </div>
            </div>

            {res.patientContext.contextNotes && (
              <p className="text-[11px] text-slate-500 italic leading-relaxed pt-1">
                {res.patientContext.contextNotes}
              </p>
            )}
          </div>
        )}

        {/* SHAP Feature Contributions (Step 10) */}
        {res.explanationAvailable ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between pb-1 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-violet-500" />
                <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                  Factors Contributing to Model Output (SHAP)
                </h3>
              </div>
              <span className="text-[11px] font-mono text-slate-500">
                Base Value: {res.shapBaseValue.toFixed(4)} ({res.explanationOutputSpace})
              </span>
            </div>

            <p className="text-[11px] text-slate-500 leading-relaxed">
              Feature contributions evaluated in log-odds space relative to the population base rate. Positive contributions push the model output toward a positive screening decision; negative contributions push the model output toward negative.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Positive Contributors */}
              <div className="space-y-2">
                <div className="flex items-center gap-1.5 text-xs font-bold text-rose-700 pb-1 border-b border-rose-100">
                  <TrendingUp className="w-3.5 h-3.5 text-rose-600" />
                  <span>Risk-Increasing Contributors ({res.positiveContributors?.length || 0})</span>
                </div>

                {(!res.positiveContributors || res.positiveContributors.length === 0) ? (
                  <p className="text-[11px] text-slate-400 py-3 text-center italic">No positive contributors</p>
                ) : (
                  res.positiveContributors.map((c, i) => (
                    <div key={i} className="p-2.5 bg-rose-50/40 rounded-xl border border-rose-100 space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-slate-800">{c.displayName || c.feature}</span>
                        <span className="font-mono text-[11px] text-slate-500">
                          {c.rawValue !== undefined && c.rawValue !== null ? String(c.rawValue) : ''}
                        </span>
                      </div>
                      <ShapBar contribution={c.contribution} maxAbsContrib={maxAbsContrib} />
                    </div>
                  ))
                )}
              </div>

              {/* Negative Contributors */}
              <div className="space-y-2">
                <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-700 pb-1 border-b border-emerald-100">
                  <TrendingDown className="w-3.5 h-3.5 text-emerald-600" />
                  <span>Risk-Decreasing Contributors ({res.negativeContributors?.length || 0})</span>
                </div>

                {(!res.negativeContributors || res.negativeContributors.length === 0) ? (
                  <p className="text-[11px] text-slate-400 py-3 text-center italic">No negative contributors</p>
                ) : (
                  res.negativeContributors.map((c, i) => (
                    <div key={i} className="p-2.5 bg-emerald-50/40 rounded-xl border border-emerald-100 space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-slate-800">{c.displayName || c.feature}</span>
                        <span className="font-mono text-[11px] text-slate-500">
                          {c.rawValue !== undefined && c.rawValue !== null ? String(c.rawValue) : ''}
                        </span>
                      </div>
                      <ShapBar contribution={c.contribution} maxAbsContrib={maxAbsContrib} />
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="p-3 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-800">
            <strong>Model explanation is currently unavailable.</strong>
            {res.explanationError && <p className="mt-1 text-[11px]">{res.explanationError}</p>}
          </div>
        )}

        {/* Model Metadata & Verification Tags */}
        <div className="p-3 bg-slate-50 border border-slate-200 rounded-xl text-[11px] text-slate-600 space-y-1 font-mono">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span>Model: {res.modelName} (v{res.modelVersion})</span>
            <span>Config: {res.configId}</span>
            <span>Calibration: {res.calibrationStatus}</span>
          </div>
          {res.assessmentId && (
            <div className="text-teal-700 pt-1 border-t border-slate-200">
              Saved to Database: Assessment ID #{res.assessmentId} · Assessed At: {formatDateTime(res.assessedAt || '')}
            </div>
          )}
        </div>

        {/* Research Disclaimer */}
        <div className="p-3 rounded-xl bg-slate-100/90 border border-slate-200 text-slate-700 text-[11px] leading-relaxed flex items-start gap-2">
          <Info className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
          <p>{res.disclaimer}</p>
        </div>
      </div>
    );
  };

  // -----------------------------------------------------------------------
  // Modal Container
  // -----------------------------------------------------------------------
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm">
      <div className="bg-white rounded-2xl max-w-3xl w-full max-h-[92vh] flex flex-col shadow-2xl border border-slate-200 animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-rose-50 text-rose-600 flex items-center justify-center">
              <Heart className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base sm:text-lg font-bold text-slate-900">
                Clinical Data Screening Assessment
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                {view === 'form' && 'Heart Disease production ML screening model (t=0.31)'}
                {view === 'review' && 'Verify clinical predictor features before running inference'}
                {view === 'loading' && 'Running inference via Spring Boot & FastAPI...'}
                {view === 'result' && 'ML inference complete · Results are decision-support only'}
                {view === 'error' && 'Assessment failed · Review errors below'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-colors"
            aria-label="Close Heart Assessment Modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="px-6 py-5 overflow-y-auto flex-1 space-y-6">
          {view === 'form' && renderForm()}
          {view === 'review' && renderReview()}
          {view === 'loading' && renderLoading()}
          {view === 'error' && renderError()}
          {view === 'result' && result && renderResult(result)}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-slate-100 flex items-center justify-between flex-shrink-0 bg-slate-50/60 rounded-b-2xl">
          {view === 'form' && (
            <>
              <span className="text-xs text-slate-400">Target variable (num) is strictly omitted.</span>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={onClose}>
                  Cancel
                </Button>
                <Button
                  type="submit"
                  form="heart-assessment-form"
                  variant="teal"
                  size="sm"
                  leftIcon={<ArrowLeft className="w-4 h-4 rotate-180" />}
                >
                  Review Inputs
                </Button>
              </div>
            </>
          )}

          {view === 'review' && (
            <>
              <Button
                variant="outline"
                size="sm"
                leftIcon={<ArrowLeft className="w-4 h-4" />}
                onClick={() => setView('form')}
              >
                Back to Form
              </Button>
              <Button
                variant="teal"
                size="sm"
                disabled={isSubmitting}
                leftIcon={<Activity className="w-4 h-4" />}
                onClick={handleConfirmAndRun}
              >
                {isSubmitting ? 'Running...' : 'Run Heart Assessment'}
              </Button>
            </>
          )}

          {view === 'result' && (
            <>
              <Button variant="outline" size="sm" onClick={handleRetry}>
                New Assessment
              </Button>
              <Button variant="teal" size="sm" onClick={onClose}>
                Close
              </Button>
            </>
          )}

          {view === 'error' && (
            <>
              <Button variant="outline" size="sm" onClick={onClose}>
                Close
              </Button>
              <Button variant="teal" size="sm" onClick={handleRetry}>
                Retry Assessment
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default HeartDiseaseAssessmentModal;
