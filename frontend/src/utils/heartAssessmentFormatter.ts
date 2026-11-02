import type { DiseaseAssessment } from '../types/assessment';
import type {
  HeartFeatureContribution,
  HeartPatientContext,
} from '../services/heartPredictionService';

/**
 * Human-readable mapping definitions for Cleveland Heart Disease clinical features.
 * In accordance with project clinical guidelines:
 * - ca represents number of major vessels colored by fluoroscopy (0-3), NOT disease stages.
 * - thal represents thallium scintigraphy defect categories.
 * - Categorical features preserve original feature name and encoded category name.
 */
export const HEART_FEATURE_LABELS = {
  sex: {
    '0': 'Female',
    '1': 'Male',
  },
  cp: {
    '1': 'Typical Angina',
    '2': 'Atypical Angina',
    '3': 'Non-Anginal Pain',
    '4': 'Asymptomatic',
  },
  fbs: {
    '0': '<= 120 mg/dL',
    '1': '> 120 mg/dL',
  },
  restecg: {
    '0': 'Normal',
    '1': 'ST-T Wave Abnormality',
    '2': 'Left Ventricular Hypertrophy',
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
    '0': '0 major vessels',
    '1': '1 major vessel',
    '2': '2 major vessels',
    '3': '3 major vessels',
  },
  thal: {
    '3': 'Normal (3.0)',
    '6': 'Fixed Defect (6.0)',
    '7': 'Reversible Defect (7.0)',
  },
} as const;

export interface OneHotFeatureStatus {
  isCategorical: boolean;
  isActive: boolean;
  transformedValue: number;
  categoryValue: number | null;
  displayStatus: string;
}

/**
 * Resolves whether a transformed one-hot encoded feature is active for the patient's
 * actual clinical input.
 *
 * One-hot encoded features in the Cleveland pipeline have the prefix 'cat__' followed by
 * <feature>_<category> (e.g., 'cat__cp_3.0', 'cat__thal_7.0', 'cat__slope_2.0').
 *
 * For a given categorical input:
 * - Exactly one category in the group is active (transformed value = 1.0).
 * - All other one-hot categories in that group are inactive (transformed value = 0.0).
 * - For reference categories dropped via drop='first' (cp=1, restecg=0, slope=1, thal=3),
 *   all one-hot dummy features in that group are inactive (0.0).
 */
export function resolveOneHotFeatureStatus(
  transformedFeature: string,
  rawFeatureValue: unknown
): OneHotFeatureStatus {
  if (!transformedFeature || !transformedFeature.startsWith('cat__')) {
    return {
      isCategorical: false,
      isActive: true,
      transformedValue: typeof rawFeatureValue === 'number' ? rawFeatureValue : 0,
      categoryValue: null,
      displayStatus: rawFeatureValue !== null && rawFeatureValue !== undefined ? String(rawFeatureValue) : 'Not provided',
    };
  }

  const parts = transformedFeature.replace(/^cat__/, '').split('_');
  const targetCatStr = parts[parts.length - 1];
  const targetCatNum = parseFloat(targetCatStr);

  const rawNum = rawFeatureValue !== null && rawFeatureValue !== undefined ? parseFloat(String(rawFeatureValue)) : NaN;

  const isActive = !isNaN(rawNum) && !isNaN(targetCatNum) && rawNum === targetCatNum;
  const transformedValue = isActive ? 1 : 0;

  return {
    isCategorical: true,
    isActive,
    transformedValue,
    categoryValue: isNaN(targetCatNum) ? null : targetCatNum,
    displayStatus: isActive ? 'Active (1)' : 'Inactive (0)',
  };
}

export interface FormattedHeartInput {
  name: string;
  label: string;
  displayValue: string;
  rawValue: unknown;
  unit?: string;
  isNullable?: boolean;
}

export interface ParsedHeartAssessment {
  id: number;
  patientId: number;
  assessedAt: string;
  probability: number;
  probabilityPercentage: string;
  threshold: number;
  prediction: number;
  isPositive: boolean;
  thresholdDescription: string;
  modelName: string;
  modelVersion: string;
  configId: string;
  outputSpace: string;
  calibrationStatus: string;
  explanationAvailable: boolean;
  explanationOutputSpace: string;
  shapBaseValue: number | null;
  modelOutput: number | null;
  positiveContributors: HeartFeatureContribution[];
  negativeContributors: HeartFeatureContribution[];
  allContributions: HeartFeatureContribution[];
  explanationError: string | null;
  inputs: FormattedHeartInput[];
  rawInputSummary: Record<string, unknown>;
  patientContext: HeartPatientContext | null;
}

/**
 * Safely parses the stored JSON in inputFeatures and reconstructs
 * real model metrics, threshold, 13 clinical inputs, patient context,
 * and SHAP feature contributions.
 *
 * Does not fabricate values or hallucinate medical diagnoses.
 */
export function parseHeartAssessment(assessment: DiseaseAssessment): ParsedHeartAssessment {
  let parsedJson: Record<string, any> = {};

  if (assessment.inputFeatures) {
    try {
      parsedJson = JSON.parse(assessment.inputFeatures);
    } catch (e) {
      console.warn('Failed to parse inputFeatures JSON for assessment #' + assessment.id, e);
      parsedJson = {};
    }
  }

  // Model-estimated probability from real record
  const probability = typeof parsedJson.probability === 'number'
    ? parsedJson.probability
    : assessment.riskScore;

  const probabilityPercentage = (probability * 100).toFixed(2) + '%';

  // Real threshold used: prefer stored threshold, fallback to default 0.31
  const threshold = typeof parsedJson.threshold === 'number'
    ? parsedJson.threshold
    : 0.31;

  // Real binary prediction: prefer stored, fallback to probability vs threshold
  const prediction = typeof parsedJson.prediction === 'number'
    ? parsedJson.prediction
    : (probability >= threshold ? 1 : 0);

  const isPositive = prediction === 1;

  const thresholdDescription = parsedJson.thresholdDescription ||
    `Configured model classification threshold (t=${threshold.toFixed(2)}) targeting >= 90% sensitivity.`;

  const modelName = parsedJson.modelName || 'Logistic Regression (Robust L1)';
  const modelVersion = parsedJson.modelVersion || '1.0.0';
  const configId = parsedJson.configId || 'LR_robust_l1_C1.0';
  const outputSpace = parsedJson.outputSpace || 'probability';
  const calibrationStatus = parsedJson.calibrationStatus || 'uncalibrated_raw_sigmoid';

  // Input summary parsing: reconstruct all 13 features
  const summary: Record<string, any> = parsedJson.inputSummary || {};

  const formatContribution = (c: HeartFeatureContribution): HeartFeatureContribution => {
    const rawVal = c.rawValue !== undefined && c.rawValue !== null ? c.rawValue : summary[c.feature];
    const status = resolveOneHotFeatureStatus(c.transformedFeature, rawVal);
    return {
      ...c,
      rawValue: rawVal,
      isCategorical: status.isCategorical,
      isActive: status.isActive,
      transformedValue: status.transformedValue,
      displayStatus: status.displayStatus,
    };
  };

  // SHAP explanations
  const rawAllContribs: HeartFeatureContribution[] = Array.isArray(parsedJson.allContributions)
    ? parsedJson.allContributions
    : [];

  const rawPositive: HeartFeatureContribution[] = Array.isArray(parsedJson.positiveContributors) && parsedJson.positiveContributors.length > 0
    ? parsedJson.positiveContributors
    : rawAllContribs.filter((c) => c.contribution > 0);

  const rawNegative: HeartFeatureContribution[] = Array.isArray(parsedJson.negativeContributors) && parsedJson.negativeContributors.length > 0
    ? parsedJson.negativeContributors
    : rawAllContribs.filter((c) => c.contribution < 0);

  const allContribs: HeartFeatureContribution[] = rawAllContribs.map(formatContribution);
  const positiveContributors: HeartFeatureContribution[] = rawPositive.map(formatContribution);
  const negativeContributors: HeartFeatureContribution[] = rawNegative.map(formatContribution);

  const explanationAvailable = Boolean(
    parsedJson.explanationAvailable !== undefined
      ? parsedJson.explanationAvailable
      : allContribs.length > 0
  );

  const explanationOutputSpace = parsedJson.explanationOutputSpace || 'log_odds';
  const shapBaseValue = typeof parsedJson.shapBaseValue === 'number' ? parsedJson.shapBaseValue : null;
  const modelOutput = typeof parsedJson.modelOutput === 'number' ? parsedJson.modelOutput : null;
  const explanationError = parsedJson.explanationError || null;

  const formatNullable = (val: unknown, map: Record<string, string>): string => {
    if (val === null || val === undefined || val === '' || val === 'null') {
      return 'Not provided';
    }
    const str = String(val).replace('.0', '');
    return map[str] || String(val);
  };

  const inputs: FormattedHeartInput[] = [
    {
      name: 'age',
      label: 'Age',
      displayValue: summary.age !== undefined && summary.age !== null ? `${summary.age} years` : 'Not provided',
      rawValue: summary.age,
      unit: 'years',
    },
    {
      name: 'sex',
      label: 'Biological Sex',
      displayValue: summary.sex !== undefined && summary.sex !== null
        ? HEART_FEATURE_LABELS.sex[String(summary.sex).replace('.0', '') as '0' | '1'] || String(summary.sex)
        : 'Not provided',
      rawValue: summary.sex,
    },
    {
      name: 'cp',
      label: 'Chest Pain Type (cp)',
      displayValue: summary.cp !== undefined && summary.cp !== null
        ? HEART_FEATURE_LABELS.cp[String(summary.cp).replace('.0', '') as '1' | '2' | '3' | '4'] || String(summary.cp)
        : 'Not provided',
      rawValue: summary.cp,
    },
    {
      name: 'trestbps',
      label: 'Resting Blood Pressure',
      displayValue: summary.trestbps !== undefined && summary.trestbps !== null ? `${summary.trestbps} mm Hg` : 'Not provided',
      rawValue: summary.trestbps,
      unit: 'mm Hg',
    },
    {
      name: 'chol',
      label: 'Serum Cholesterol',
      displayValue: summary.chol !== undefined && summary.chol !== null ? `${summary.chol} mg/dL` : 'Not provided',
      rawValue: summary.chol,
      unit: 'mg/dL',
    },
    {
      name: 'fbs',
      label: 'Fasting Blood Sugar',
      displayValue: summary.fbs !== undefined && summary.fbs !== null
        ? HEART_FEATURE_LABELS.fbs[String(summary.fbs).replace('.0', '') as '0' | '1'] || String(summary.fbs)
        : 'Not provided',
      rawValue: summary.fbs,
    },
    {
      name: 'restecg',
      label: 'Resting ECG',
      displayValue: summary.restecg !== undefined && summary.restecg !== null
        ? HEART_FEATURE_LABELS.restecg[String(summary.restecg).replace('.0', '') as '0' | '1' | '2'] || String(summary.restecg)
        : 'Not provided',
      rawValue: summary.restecg,
    },
    {
      name: 'thalach',
      label: 'Maximum Heart Rate (thalach)',
      displayValue: summary.thalach !== undefined && summary.thalach !== null ? `${summary.thalach} bpm` : 'Not provided',
      rawValue: summary.thalach,
      unit: 'bpm',
    },
    {
      name: 'exang',
      label: 'Exercise-Induced Angina',
      displayValue: summary.exang !== undefined && summary.exang !== null
        ? HEART_FEATURE_LABELS.exang[String(summary.exang).replace('.0', '') as '0' | '1'] || String(summary.exang)
        : 'Not provided',
      rawValue: summary.exang,
    },
    {
      name: 'oldpeak',
      label: 'ST Depression (oldpeak)',
      displayValue: summary.oldpeak !== undefined && summary.oldpeak !== null ? `${summary.oldpeak} mm` : 'Not provided',
      rawValue: summary.oldpeak,
      unit: 'mm',
    },
    {
      name: 'slope',
      label: 'ST Segment Slope',
      displayValue: summary.slope !== undefined && summary.slope !== null
        ? HEART_FEATURE_LABELS.slope[String(summary.slope).replace('.0', '') as '1' | '2' | '3'] || String(summary.slope)
        : 'Not provided',
      rawValue: summary.slope,
    },
    {
      name: 'ca',
      label: 'Major Vessels Colored (ca)',
      displayValue: formatNullable(summary.ca, HEART_FEATURE_LABELS.ca),
      rawValue: summary.ca,
      isNullable: true,
    },
    {
      name: 'thal',
      label: 'Thallium Scintigraphy (thal)',
      displayValue: formatNullable(summary.thal, HEART_FEATURE_LABELS.thal),
      rawValue: summary.thal,
      isNullable: true,
    },
  ];

  // Patient context
  const patientContext: HeartPatientContext | null = parsedJson.patientContext || null;

  return {
    id: assessment.id,
    patientId: assessment.patientId,
    assessedAt: assessment.assessedAt,
    probability,
    probabilityPercentage,
    threshold,
    prediction,
    isPositive,
    thresholdDescription,
    modelName,
    modelVersion,
    configId,
    outputSpace,
    calibrationStatus,
    explanationAvailable,
    explanationOutputSpace,
    shapBaseValue,
    modelOutput,
    positiveContributors,
    negativeContributors,
    allContributions: allContribs,
    explanationError,
    inputs,
    rawInputSummary: summary,
    patientContext,
  };
}
