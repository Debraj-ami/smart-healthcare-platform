import React, { useState, useEffect } from 'react';
import {
  X,
  Activity,
  ShieldAlert,
  AlertCircle,
  ArrowLeft,
  Info,
  Loader2,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  Minus,
  FlaskConical,
  TriangleAlert,
} from 'lucide-react';
import { LiverDiseaseFeatures } from '../../types/assessment';
import Button from '../common/Button';
import { usePatient } from '../../context/PatientContext';
import liverPredictionService, {
  LiverPredictResponse,
  FeatureContribution,
} from '../../services/liverPredictionService';

interface LiverDiseaseAssessmentModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAssessmentComplete?: () => void;
  defaultAge?: number | null;
  defaultGender?: string | null;
}

const INITIAL_FORM: LiverDiseaseFeatures = {
  age: '',
  gender: '',
  totalBilirubin: '',
  directBilirubin: '',
  alkalinePhosphatase: '',
  sgpt: '',
  sgot: '',
  totalProteins: '',
  albumin: '',
  agRatio: '',
};

type FormErrors = Partial<Record<keyof LiverDiseaseFeatures, string>>;

type ModalView = 'form' | 'loading' | 'result' | 'error';



// -----------------------------------------------------------------------
// SHAP direction badge
// -----------------------------------------------------------------------

function DirectionBadge({ direction }: { direction: string }) {
  if (direction === 'INCREASES_RISK') {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-red-100 text-red-700">
        <TrendingUp className="w-2.5 h-2.5" /> Risk↑
      </span>
    );
  }
  if (direction === 'DECREASES_RISK') {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-emerald-100 text-emerald-700">
        <TrendingDown className="w-2.5 h-2.5" /> Risk↓
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-slate-100 text-slate-600">
      <Minus className="w-2.5 h-2.5" /> Neutral
    </span>
  );
}

// -----------------------------------------------------------------------
// SHAP contribution bar
// -----------------------------------------------------------------------

function ShapBar({ contribution, maxAbsContrib }: { contribution: number; maxAbsContrib: number }) {
  const pct = maxAbsContrib > 0 ? Math.abs(contribution) / maxAbsContrib : 0;
  const width = Math.round(pct * 100);
  const color = contribution > 0 ? 'bg-red-400' : contribution < 0 ? 'bg-emerald-400' : 'bg-slate-300';
  return (
    <div className="flex items-center gap-1.5 w-full">
      <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${width}%` }} />
      </div>
      <span className="text-[10px] font-mono text-slate-500 w-12 text-right">
        {contribution > 0 ? '+' : ''}{contribution.toFixed(4)}
      </span>
    </div>
  );
}

// -----------------------------------------------------------------------
// Main modal component
// -----------------------------------------------------------------------

export const LiverDiseaseAssessmentModal: React.FC<LiverDiseaseAssessmentModalProps> = ({
  isOpen,
  onClose,
  onAssessmentComplete,
  defaultAge,
  defaultGender,
}) => {
  const { activePatientId } = usePatient();
  const [formData, setFormData] = useState<LiverDiseaseFeatures>(INITIAL_FORM);
  const [errors, setErrors] = useState<FormErrors>({});
  const [view, setView] = useState<ModalView>('form');
  const [result, setResult] = useState<LiverPredictResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');

  useEffect(() => {
    if (isOpen) {
      setFormData({
        ...INITIAL_FORM,
        age: defaultAge ? String(defaultAge) : '',
        gender:
          defaultGender && (defaultGender.toUpperCase() === 'MALE' || defaultGender.toUpperCase() === 'FEMALE')
            ? (defaultGender.charAt(0).toUpperCase() + defaultGender.slice(1).toLowerCase() as 'Male' | 'Female')
            : '',
      });
      setErrors({});
      setView('form');
      setResult(null);
      setErrorMessage('');
    }
  }, [isOpen, defaultAge, defaultGender]);

  if (!isOpen) return null;

  const handleChange = (field: keyof LiverDiseaseFeatures, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    if (errors[field]) setErrors((prev) => ({ ...prev, [field]: undefined }));
  };

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    const ageNum = Number(formData.age);
    if (!formData.age.trim()) newErrors.age = 'Age is required';
    else if (isNaN(ageNum) || !Number.isInteger(ageNum) || ageNum < 1 || ageNum > 120)
      newErrors.age = 'Must be an integer between 1 and 120';

    if (!formData.gender) newErrors.gender = 'Please select gender';
    else if (formData.gender !== 'Male' && formData.gender !== 'Female')
      newErrors.gender = 'Must be Male or Female';

    const tbNum = Number(formData.totalBilirubin);
    if (!formData.totalBilirubin.trim()) newErrors.totalBilirubin = 'Total Bilirubin is required';
    else if (isNaN(tbNum) || tbNum <= 0 || tbNum > 100) newErrors.totalBilirubin = 'Valid positive value (0.1–100 mg/dL)';

    const dbNum = Number(formData.directBilirubin);
    if (!formData.directBilirubin.trim()) newErrors.directBilirubin = 'Direct Bilirubin is required';
    else if (isNaN(dbNum) || dbNum < 0 || dbNum > 50) newErrors.directBilirubin = 'Valid non-negative value (0.0–50 mg/dL)';

    const alkNum = Number(formData.alkalinePhosphatase);
    if (!formData.alkalinePhosphatase.trim()) newErrors.alkalinePhosphatase = 'Alkaline Phosphatase is required';
    else if (isNaN(alkNum) || alkNum <= 0 || alkNum > 5000) newErrors.alkalinePhosphatase = 'Valid positive value (IU/L)';

    const sgptNum = Number(formData.sgpt);
    if (!formData.sgpt.trim()) newErrors.sgpt = 'SGPT (ALT) is required';
    else if (isNaN(sgptNum) || sgptNum <= 0 || sgptNum > 5000) newErrors.sgpt = 'Valid positive value (U/L)';

    const sgotNum = Number(formData.sgot);
    if (!formData.sgot.trim()) newErrors.sgot = 'SGOT (AST) is required';
    else if (isNaN(sgotNum) || sgotNum <= 0 || sgotNum > 5000) newErrors.sgot = 'Valid positive value (U/L)';

    const tpNum = Number(formData.totalProteins);
    if (!formData.totalProteins.trim()) newErrors.totalProteins = 'Total Proteins is required';
    else if (isNaN(tpNum) || tpNum <= 0 || tpNum > 20) newErrors.totalProteins = 'Valid positive value (1.0–15.0 g/dL)';

    const albNum = Number(formData.albumin);
    if (!formData.albumin.trim()) newErrors.albumin = 'Albumin is required';
    else if (isNaN(albNum) || albNum <= 0 || albNum > 15) newErrors.albumin = 'Valid positive value (0.5–10.0 g/dL)';

    const agNum = Number(formData.agRatio);
    if (!formData.agRatio.trim()) newErrors.agRatio = 'A/G Ratio is required';
    else if (isNaN(agNum) || agNum <= 0 || agNum > 10) newErrors.agRatio = 'Valid positive ratio (0.1–5.0)';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleRunAssessment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateForm()) return;

    setView('loading');

    try {
      const prediction = await liverPredictionService.predict({
        patientId: activePatientId,
        age: parseInt(formData.age, 10),
        gender: formData.gender as 'Male' | 'Female',
        totalBilirubin: parseFloat(formData.totalBilirubin),
        directBilirubin: parseFloat(formData.directBilirubin),
        alkalinePhosphatase: parseFloat(formData.alkalinePhosphatase),
        sgpt: parseFloat(formData.sgpt),
        sgot: parseFloat(formData.sgot),
        totalProteins: parseFloat(formData.totalProteins),
        albumin: parseFloat(formData.albumin),
        agRatio: parseFloat(formData.agRatio),
      });
      setResult(prediction);
      setView('result');
      // Notify parent to refresh assessment history
      if (onAssessmentComplete) {
        onAssessmentComplete();
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'An unexpected error occurred.';
      setErrorMessage(msg);
      setView('error');
    }
  };

  const handleRetry = () => {
    setView('form');
    setErrors({});
    setErrorMessage('');
  };

  // -----------------------------------------------------------------------
  // Render helpers
  // -----------------------------------------------------------------------

  const renderInputField = (
    field: keyof LiverDiseaseFeatures,
    label: string,
    unit: string,
    placeholder: string,
    step: string,
    min: string
  ) => (
    <div>
      <label className="block text-xs font-semibold text-slate-700 mb-1">
        {label} <span className="text-rose-500">*</span>
      </label>
      <div className="relative">
        <input
          type="number"
          step={step}
          min={min}
          value={formData[field]}
          onChange={(e) => handleChange(field, e.target.value)}
          placeholder={placeholder}
          className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
            errors[field]
              ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
              : 'border-slate-200 focus:ring-teal-500/20 focus:border-teal-500'
          }`}
        />
        <span className="absolute right-3 top-2.5 text-xs text-slate-400">{unit}</span>
      </div>
      {errors[field] && (
        <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
          <AlertCircle className="w-3 h-3 flex-shrink-0" />
          {errors[field]}
        </p>
      )}
    </div>
  );

  const renderForm = () => (
    <form id="liver-assessment-form" onSubmit={handleRunAssessment} className="space-y-6">
      {/* Section 1: Patient Information */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 pb-1 border-b border-slate-100">
          <span className="w-2 h-2 rounded-full bg-teal-500" />
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Patient Information</h3>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Age <span className="text-rose-500">*</span>
            </label>
            <div className="relative">
              <input
                type="number"
                min="1"
                max="120"
                step="1"
                value={formData.age}
                onChange={(e) => handleChange('age', e.target.value)}
                placeholder="e.g. 45"
                className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                  errors.age
                    ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                    : 'border-slate-200 focus:ring-teal-500/20 focus:border-teal-500'
                }`}
              />
              <span className="absolute right-3 top-2.5 text-xs text-slate-400">years</span>
            </div>
            {errors.age && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />{errors.age}
              </p>
            )}
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Gender <span className="text-rose-500">*</span>
            </label>
            <select
              value={formData.gender}
              onChange={(e) => handleChange('gender', e.target.value as 'Male' | 'Female' | '')}
              className={`w-full px-3 py-2 bg-slate-50 border rounded-xl text-sm focus:outline-none focus:ring-2 ${
                errors.gender
                  ? 'border-rose-300 focus:ring-rose-500/20 focus:border-rose-500 bg-rose-50/20'
                  : 'border-slate-200 focus:ring-teal-500/20 focus:border-teal-500'
              }`}
            >
              <option value="">Select Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
            {errors.gender && (
              <p className="text-[11px] text-rose-600 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />{errors.gender}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Section 2: Liver Function Measurements */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 pb-1 border-b border-slate-100">
          <span className="w-2 h-2 rounded-full bg-amber-500" />
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Liver Function Measurements</h3>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {renderInputField('totalBilirubin', 'Total Bilirubin', 'mg/dL', 'e.g. 0.9', '0.1', '0.1')}
          {renderInputField('directBilirubin', 'Direct Bilirubin', 'mg/dL', 'e.g. 0.3', '0.1', '0.0')}
          {renderInputField('alkalinePhosphatase', 'Alkaline Phosphatase', 'IU/L', 'e.g. 200', '1', '1')}
          {renderInputField('sgpt', 'SGPT (ALT)', 'U/L', 'e.g. 35', '1', '1')}
          {renderInputField('sgot', 'SGOT (AST)', 'U/L', 'e.g. 40', '1', '1')}
        </div>
      </div>

      {/* Section 3: Protein Measurements */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 pb-1 border-b border-slate-100">
          <span className="w-2 h-2 rounded-full bg-blue-500" />
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">Protein Measurements</h3>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {renderInputField('totalProteins', 'Total Proteins', 'g/dL', 'e.g. 6.8', '0.1', '0.1')}
          {renderInputField('albumin', 'Albumin', 'g/dL', 'e.g. 3.4', '0.1', '0.1')}
          {renderInputField('agRatio', 'A/G Ratio', 'ratio', 'e.g. 1.0', '0.01', '0.01')}
        </div>
      </div>
    </form>
  );

  const renderLoading = () => (
    <div className="flex flex-col items-center justify-center py-16 gap-4">
      <div className="relative">
        <div className="w-16 h-16 rounded-full bg-amber-50 flex items-center justify-center">
          <FlaskConical className="w-8 h-8 text-amber-500" />
        </div>
        <Loader2 className="absolute -top-1 -right-1 w-6 h-6 text-teal-500 animate-spin" />
      </div>
      <div className="text-center space-y-1">
        <h4 className="text-sm font-bold text-slate-800">Running ML Inference</h4>
        <p className="text-xs text-slate-500 max-w-xs leading-relaxed">
          Sending feature inputs to the Logistic Regression model (Isotonic Calibrated).
          Computing calibrated probability and SHAP local feature attributions…
        </p>
      </div>
    </div>
  );

  const renderError = () => (
    <div className="space-y-4 py-4">
      <div className="p-4 rounded-xl bg-red-50 border border-red-200 flex items-start gap-3">
        <TriangleAlert className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="text-sm font-bold text-red-900">Prediction Failed</h4>
          <p className="text-xs text-red-700 leading-relaxed font-mono break-all">{errorMessage}</p>
        </div>
      </div>
      <div className="p-3 rounded-xl bg-blue-50 border border-blue-200 text-blue-800 text-xs space-y-1">
        <p className="font-semibold">Troubleshooting:</p>
        <ol className="list-decimal list-inside space-y-0.5 text-blue-700">
          <li>Ensure the Python ML service is running on port 8001</li>
          <li>
            From the project root: <code className="font-mono bg-blue-100 px-1 rounded">cd ml-service && uvicorn app.main:app --port 8001</code>
          </li>
          <li>Ensure Spring Boot is running on port 8080</li>
        </ol>
      </div>
    </div>
  );

  const renderResult = (res: LiverPredictResponse) => {
    const maxAbsContrib = res.shapFeatures
      ? Math.max(...res.shapFeatures.map((f) => Math.abs(f.contribution)), 0.0001)
      : 0.0001;

    const probPct = Math.round(res.probability * 100);

    return (
      <div className="space-y-5 animate-in fade-in duration-200">
        {/* Probability Output */}
        <div className="p-4 rounded-xl border bg-teal-50/70 border-teal-200 text-teal-950 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-bold text-teal-950">Model Output — Calibrated Probability</h4>
              <p className="text-xs text-teal-700 mt-0.5">Continuous ML Risk Estimate</p>
            </div>
            <CheckCircle2 className="w-5 h-5 text-teal-600 flex-shrink-0" />
          </div>

          {/* Probability bar */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
              <span>P(Liver Disease)</span>
              <span className="text-lg font-extrabold text-teal-700">{probPct}%</span>
            </div>
            <div className="h-2.5 bg-teal-100 rounded-full overflow-hidden border border-teal-200/60">
              <div
                className="h-full rounded-full transition-all duration-700 bg-teal-600"
                style={{ width: `${probPct}%` }}
              />
            </div>
            <p className="text-[11px] text-teal-800/80">
              Raw probability: {res.probability.toFixed(4)} · Model: {res.modelName} · Calibration: {res.calibrationMethod}
            </p>
          </div>
        </div>

        {/* Research Disclaimer */}
        <div className="p-3 rounded-xl bg-slate-100/80 border border-slate-200 text-slate-700 text-[11px] leading-relaxed flex items-start gap-2">
          <Info className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
          <p>{res.disclaimer}</p>
        </div>

        {/* SHAP Explanations */}
        {res.shapFeatures && res.shapFeatures.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 pb-1 border-b border-slate-100">
              <span className="w-2 h-2 rounded-full bg-violet-500" />
              <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                Local Feature Contributions (SHAP)
              </h3>
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed">
              SHAP values show each feature's contribution to the model output in log-odds space.
              Base value: <span className="font-mono">{res.shapBaseValue?.toFixed(4)}</span> ({res.shapOutputSpace}).
              Contributions explain model behavior, not biological causation.
            </p>

            <div className="space-y-2">
              {(res.shapFeatures as FeatureContribution[])
                .slice()
                .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                .map((feat, idx) => (
                  <div
                    key={idx}
                    className="bg-slate-50 rounded-xl border border-slate-100 px-3 py-2.5 space-y-1.5"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-800">{feat.feature}</span>
                        <DirectionBadge direction={feat.direction} />
                      </div>
                      <span className="text-[11px] text-slate-500 font-mono">
                        {typeof feat.value === 'number' ? feat.value.toLocaleString() : feat.value}
                      </span>
                    </div>
                    <ShapBar contribution={feat.contribution} maxAbsContrib={maxAbsContrib} />
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Assessment ID / Persistence confirmation */}
        <div className="p-3 rounded-xl bg-teal-50 border border-teal-200 text-teal-800 text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-teal-600 flex-shrink-0" />
          <span>
            Assessment saved to database — ID #{res.assessmentId} · Patient #{res.patientId}
          </span>
        </div>
      </div>
    );
  };

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm">
      <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[92vh] flex flex-col shadow-2xl border border-slate-200 animate-in fade-in zoom-in-95 duration-150">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base sm:text-lg font-bold text-slate-900">
                Liver Disease Risk Assessment
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                {view === 'result'
                  ? 'ML inference complete · Results are research-grade only'
                  : 'Enter patient measurements for calibrated ML risk screening'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 overflow-y-auto flex-1 space-y-6">
          {view === 'form' && renderForm()}
          {view === 'loading' && renderLoading()}
          {view === 'error' && renderError()}
          {view === 'result' && result && renderResult(result)}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-100 flex items-center justify-between flex-shrink-0 bg-slate-50/60 rounded-b-2xl">
          {view === 'form' && (
            <>
              <span className="text-xs text-slate-400">All 10 biometric/lab features are required.</span>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={onClose}>Cancel</Button>
                <Button
                  type="submit"
                  form="liver-assessment-form"
                  variant="teal"
                  size="sm"
                  leftIcon={<Activity className="w-4 h-4" />}
                >
                  Run Assessment
                </Button>
              </div>
            </>
          )}

          {view === 'loading' && (
            <>
              <span className="text-xs text-slate-400">Waiting for ML inference…</span>
              <Button variant="outline" size="sm" disabled>Cancel</Button>
            </>
          )}

          {view === 'error' && (
            <>
              <span className="text-xs text-slate-400">ML service unreachable or returned an error.</span>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={onClose}>Close</Button>
                <Button variant="teal" size="sm" onClick={handleRetry} leftIcon={<ArrowLeft className="w-4 h-4" />}>
                  Edit Inputs
                </Button>
              </div>
            </>
          )}

          {view === 'result' && (
            <>
              <span className="text-xs text-slate-400">Result persisted · ID #{result?.assessmentId}</span>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={handleRetry} leftIcon={<ArrowLeft className="w-4 h-4" />}>
                  New Assessment
                </Button>
                <Button variant="teal" size="sm" onClick={onClose}>Done</Button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default LiverDiseaseAssessmentModal;
