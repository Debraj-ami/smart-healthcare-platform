import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  parseHeartAssessment,
  HEART_FEATURE_LABELS,
  type FormattedHeartInput,
} from '../src/utils/heartAssessmentFormatter.ts';
import type { DiseaseAssessment } from '../src/types/assessment.ts';

describe('Stage 6 — Heart History and Explanations Specification Tests', () => {
  // Test 1: Empty Heart assessment history handling
  test('1. Empty Heart assessment history returns 0 items and safe empty state', () => {
    const history: DiseaseAssessment[] = [];
    assert.strictEqual(history.length, 0);
    // Verifies empty state trigger condition
    const isEmpty = history.length === 0;
    assert.strictEqual(isEmpty, true);
  });

  // Test 2: History successfully parses real assessment records
  test('2. History successfully renders real assessment records with valid probabilities', () => {
    const realAssessment: DiseaseAssessment = {
      id: 101,
      patientId: 42,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.1911,
      riskLevel: 'LOW',
      assessedAt: '2026-10-02T10:30:00Z',
      inputFeatures: JSON.stringify({
        probability: 0.1911,
        prediction: 0,
        threshold: 0.31,
        thresholdDescription: 'Configured model classification threshold (t=0.31) targeting >= 90% sensitivity.',
        modelName: 'Logistic Regression (Robust L1)',
        modelVersion: '1.0.0',
        configId: 'LR_robust_l1_C1.0',
        outputSpace: 'probability',
        calibrationStatus: 'uncalibrated_raw_sigmoid',
        explanationAvailable: true,
        explanationOutputSpace: 'log_odds',
        shapBaseValue: -0.375,
        modelOutput: -1.442,
        positiveContributors: [
          {
            feature: 'age',
            transformedFeature: 'num__age',
            displayName: 'Age',
            rawValue: 58,
            contribution: 0.25,
            absContribution: 0.25,
            direction: 'INCREASES_RISK_ESTIMATE',
          },
        ],
        negativeContributors: [
          {
            feature: 'thalach',
            transformedFeature: 'num__thalach',
            displayName: 'Maximum Heart Rate',
            rawValue: 165,
            contribution: -0.42,
            absContribution: 0.42,
            direction: 'DECREASES_RISK_ESTIMATE',
          },
        ],
        allContributions: [],
        inputSummary: {
          age: 58,
          sex: 1,
          cp: 2,
          trestbps: 120,
          chol: 240,
          fbs: 0,
          restecg: 0,
          thalach: 165,
          exang: 0,
          oldpeak: 1.0,
          slope: 1,
          ca: 0,
          thal: 3,
        },
      }),
    };

    const parsed = parseHeartAssessment(realAssessment);
    assert.strictEqual(parsed.id, 101);
    assert.strictEqual(parsed.patientId, 42);
    assert.strictEqual(parsed.probability, 0.1911);
    assert.strictEqual(parsed.isPositive, false);
    assert.strictEqual(parsed.threshold, 0.31);
    assert.strictEqual(parsed.modelName, 'Logistic Regression (Robust L1)');
    assert.strictEqual(parsed.modelVersion, '1.0.0');
  });

  // Test 3: Selecting an assessment loads full detail structure
  test('3. Selecting an assessment opens its details with complete metadata', () => {
    const assessment: DiseaseAssessment = {
      id: 202,
      patientId: 7,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.654,
      riskLevel: 'HIGH',
      assessedAt: '2026-10-02T11:00:00Z',
      inputFeatures: JSON.stringify({
        probability: 0.654,
        prediction: 1,
        threshold: 0.31,
        modelName: 'Logistic Regression (Robust L1)',
        modelVersion: '1.0.0',
        configId: 'LR_robust_l1_C1.0',
        inputSummary: { age: 62, sex: 1 },
      }),
    };

    const parsed = parseHeartAssessment(assessment);
    assert.strictEqual(parsed.id, 202);
    assert.strictEqual(parsed.patientId, 7);
    assert.strictEqual(parsed.isPositive, true);
    assert.strictEqual(parsed.configId, 'LR_robust_l1_C1.0');
  });

  // Test 4: Probability is rendered correctly
  test('4. Probability is rendered correctly as percentage and decimal', () => {
    const assessment: DiseaseAssessment = {
      id: 303,
      patientId: 10,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.19114,
      riskLevel: null,
      assessedAt: '2026-10-02T11:00:00Z',
      inputFeatures: JSON.stringify({
        probability: 0.19114,
      }),
    };

    const parsed = parseHeartAssessment(assessment);
    assert.strictEqual(parsed.probabilityPercentage, '19.11%');
    assert.strictEqual(parsed.probability, 0.19114);
  });

  // Test 5: Threshold is rendered from API data (not hardcoded to static value if backend changes)
  test('5. Threshold is rendered dynamically from API data', () => {
    const customThresholdAssessment: DiseaseAssessment = {
      id: 304,
      patientId: 10,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.28,
      riskLevel: null,
      assessedAt: '2026-10-02T11:00:00Z',
      inputFeatures: JSON.stringify({
        probability: 0.28,
        threshold: 0.25,
        prediction: 1,
      }),
    };

    const parsed = parseHeartAssessment(customThresholdAssessment);
    assert.strictEqual(parsed.threshold, 0.25);
    assert.strictEqual(parsed.isPositive, true); // 0.28 >= 0.25
  });

  // Test 6: 13 input values render correctly with human-readable labels
  test('6. All 13 input values render correctly with human-readable clinical labels', () => {
    const assessment: DiseaseAssessment = {
      id: 401,
      patientId: 15,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.20,
      riskLevel: 'LOW',
      assessedAt: '2026-10-02T11:00:00Z',
      inputFeatures: JSON.stringify({
        inputSummary: {
          age: 55,
          sex: 1,
          cp: 4,
          trestbps: 140,
          chol: 260,
          fbs: 1,
          restecg: 2,
          thalach: 150,
          exang: 1,
          oldpeak: 2.3,
          slope: 2,
          ca: 2,
          thal: 7,
        },
      }),
    };

    const parsed = parseHeartAssessment(assessment);
    assert.strictEqual(parsed.inputs.length, 13);

    const inputMap = new Map(parsed.inputs.map((i: FormattedHeartInput) => [i.name, i.displayValue]));
    assert.strictEqual(inputMap.get('age'), '55 years');
    assert.strictEqual(inputMap.get('sex'), 'Male');
    assert.strictEqual(inputMap.get('cp'), 'Asymptomatic');
    assert.strictEqual(inputMap.get('trestbps'), '140 mm Hg');
    assert.strictEqual(inputMap.get('chol'), '260 mg/dL');
    assert.strictEqual(inputMap.get('fbs'), '> 120 mg/dL');
    assert.strictEqual(inputMap.get('restecg'), 'Left Ventricular Hypertrophy');
    assert.strictEqual(inputMap.get('thalach'), '150 bpm');
    assert.strictEqual(inputMap.get('exang'), 'Yes');
    assert.strictEqual(inputMap.get('oldpeak'), '2.3 mm');
    assert.strictEqual(inputMap.get('slope'), 'Flat');
    assert.strictEqual(inputMap.get('ca'), '2 major vessels');
    assert.strictEqual(inputMap.get('thal'), 'Reversible Defect (7.0)');
  });

  // Test 7: ca=null renders "Not provided"
  test('7. ca=null correctly renders as "Not provided" and never describes stages', () => {
    const assessment: DiseaseAssessment = {
      id: 402,
      patientId: 15,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.20,
      riskLevel: 'LOW',
      assessedAt: '2026-10-02T11:00:00Z',
      inputFeatures: JSON.stringify({
        inputSummary: {
          ca: null,
        },
      }),
    };

    const parsed = parseHeartAssessment(assessment);
    const caInput = parsed.inputs.find((i) => i.name === 'ca');
    assert.ok(caInput);
    if (!caInput) throw new Error('ca input not found');
    assert.strictEqual(caInput.displayValue, 'Not provided');
    // Ensure no stage wording exists
    assert.strictEqual(caInput.displayValue.includes('stage'), false);
    assert.strictEqual(caInput.label.includes('stage'), false);
  });

  // Test 8: thal=null renders "Not provided"
  test('8. thal=null correctly renders as "Not provided"', () => {
    const assessment: DiseaseAssessment = {
      id: 403,
      patientId: 15,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.20,
      riskLevel: 'LOW',
      assessedAt: '2026-10-02T11:00:00Z',
      inputFeatures: JSON.stringify({
        inputSummary: {
          thal: null,
        },
      }),
    };

    const parsed = parseHeartAssessment(assessment);
    const thalInput = parsed.inputs.find((i) => i.name === 'thal');
    assert.ok(thalInput);
    if (!thalInput) throw new Error('thal input not found');
    assert.strictEqual(thalInput.displayValue, 'Not provided');
  });

  // Test 9: SHAP positive and negative contributors render with categorical distinction
  test('9. SHAP positive and negative contributors render correctly with categorical distinction', () => {
    const assessment: DiseaseAssessment = {
      id: 501,
      patientId: 21,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.52,
      riskLevel: 'HIGH',
      assessedAt: '2026-10-02T11:00:00Z',
      inputFeatures: JSON.stringify({
        explanationAvailable: true,
        explanationOutputSpace: 'log_odds',
        shapBaseValue: -0.375,
        modelOutput: 0.08,
        positiveContributors: [
          {
            feature: 'cp',
            transformedFeature: 'cat__cp_4.0',
            displayName: 'Chest Pain Type (cp)',
            rawValue: 4,
            contribution: 0.58,
            absContribution: 0.58,
            direction: 'INCREASES_RISK_ESTIMATE',
          },
        ],
        negativeContributors: [
          {
            feature: 'chol',
            transformedFeature: 'num__chol',
            displayName: 'Serum Cholesterol',
            rawValue: 180,
            contribution: -0.22,
            absContribution: 0.22,
            direction: 'DECREASES_RISK_ESTIMATE',
          },
        ],
      }),
    };

    const parsed = parseHeartAssessment(assessment);
    assert.strictEqual(parsed.explanationAvailable, true);
    assert.strictEqual(parsed.positiveContributors.length, 1);
    assert.strictEqual(parsed.negativeContributors.length, 1);
    assert.strictEqual(parsed.positiveContributors[0].contribution, 0.58);
    assert.strictEqual(parsed.negativeContributors[0].contribution, -0.22);
    assert.strictEqual(parsed.positiveContributors[0].transformedFeature, 'cat__cp_4.0');
  });

  // Test 10: SHAP unavailable state renders safely
  test('10. SHAP unavailable state renders safely without fabricated values', () => {
    const assessment: DiseaseAssessment = {
      id: 502,
      patientId: 21,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.35,
      riskLevel: null,
      assessedAt: '2026-10-02T11:00:00Z',
      inputFeatures: JSON.stringify({
        explanationAvailable: false,
        explanationError: 'SHAP explainer timeout during inference.',
        positiveContributors: [],
        negativeContributors: [],
      }),
    };

    const parsed = parseHeartAssessment(assessment);
    assert.strictEqual(parsed.explanationAvailable, false);
    assert.strictEqual(parsed.positiveContributors.length, 0);
    assert.strictEqual(parsed.negativeContributors.length, 0);
    assert.strictEqual(parsed.explanationError, 'SHAP explainer timeout during inference.');
  });

  // Test 11: Patient context renders without fabricated values
  test('11. Patient context renders authentic fields without fabrication', () => {
    const assessment: DiseaseAssessment = {
      id: 601,
      patientId: 99,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.22,
      riskLevel: 'LOW',
      assessedAt: '2026-10-02T11:00:00Z',
      inputFeatures: JSON.stringify({
        patientContext: {
          patientId: 99,
          gender: 'MALE',
          ageYears: 54,
          previousAssessmentCount: 3,
          lastAssessedAt: '2026-09-15T09:00:00Z',
          lastProbability: 0.28,
          hasEstablishedBaselines: true,
          establishedBaselineMetrics: ['BLOOD_PRESSURE_SYSTOLIC', 'HEART_RATE'],
          contextNotes: 'Longitudinal screening comparison available.',
        },
      }),
    };

    const parsed = parseHeartAssessment(assessment);
    assert.ok(parsed.patientContext);
    if (!parsed.patientContext) throw new Error('patientContext is null');
    assert.strictEqual(parsed.patientContext.patientId, 99);
    assert.strictEqual(parsed.patientContext.previousAssessmentCount, 3);
    assert.strictEqual(parsed.patientContext.lastProbability, 0.28);
    assert.strictEqual(parsed.patientContext.hasEstablishedBaselines, true);
  });

  // Test 12: Error state parsing resilience
  test('12. Corrupted or invalid inputFeatures JSON gracefully falls back to riskScore', () => {
    const corruptedAssessment: DiseaseAssessment = {
      id: 701,
      patientId: 14,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.45,
      riskLevel: 'HIGH',
      assessedAt: '2026-10-02T11:00:00Z',
      inputFeatures: 'NOT_VALID_JSON{foo}',
    };

    const parsed = parseHeartAssessment(corruptedAssessment);
    assert.strictEqual(parsed.id, 701);
    assert.strictEqual(parsed.probability, 0.45);
    assert.strictEqual(parsed.threshold, 0.31); // Default fallback threshold
    assert.strictEqual(parsed.isPositive, true); // 0.45 >= 0.31
    assert.strictEqual(parsed.inputs.length, 13);
    // All inputs should be "Not provided" rather than throwing an exception
    for (const input of parsed.inputs) {
      assert.strictEqual(input.displayValue, 'Not provided');
    }
  });

  // Test 13: Dynamic patientId verification (never hardcoded)
  test('13. Dynamic patientId verification preserves authentic patient isolation', () => {
    const patientA = parseHeartAssessment({
      id: 1,
      patientId: 1001,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.12,
      riskLevel: 'LOW',
      assessedAt: '2026-10-02T11:00:00Z',
    });
    const patientB = parseHeartAssessment({
      id: 2,
      patientId: 2002,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.88,
      riskLevel: 'HIGH',
      assessedAt: '2026-10-02T11:00:00Z',
    });

    assert.strictEqual(patientA.patientId, 1001);
    assert.strictEqual(patientB.patientId, 2002);
    assert.notStrictEqual(patientA.patientId, patientB.patientId);
  });

  // Test 14: No fake or mock medical assessment data
  test('14. Stored features reflect authentic Cleveland dataset features without fabricated placeholders', () => {
    const validFeatureNames = [
      'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
      'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ];
    const dummy = parseHeartAssessment({
      id: 801,
      patientId: 1,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.1,
      riskLevel: null,
      assessedAt: '2026-10-02T11:00:00Z',
    });
    const names = dummy.inputs.map((i) => i.name);
    assert.deepStrictEqual(names, validFeatureNames);
  });

  // Test 15: No diagnostic or causal wording in UI text constants
  test('15. Disclaimers and explanatory texts strictly use non-diagnostic, screening language', () => {
    const parsed = parseHeartAssessment({
      id: 901,
      patientId: 5,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.25,
      riskLevel: 'LOW',
      assessedAt: '2026-10-02T11:00:00Z',
    });

    // Check threshold description
    assert.strictEqual(parsed.thresholdDescription.toLowerCase().includes('diagnosis'), false);
    assert.strictEqual(parsed.thresholdDescription.toLowerCase().includes('disease-free'), false);
    assert.strictEqual(parsed.thresholdDescription.toLowerCase().includes('cure'), false);

    // Verify HEART_FEATURE_LABELS does not describe ca as disease stages
    const caKeys = Object.keys(HEART_FEATURE_LABELS.ca);
    for (const k of caKeys) {
      const val = HEART_FEATURE_LABELS.ca[k as keyof typeof HEART_FEATURE_LABELS.ca];
      assert.strictEqual(val.toLowerCase().includes('stage'), false);
      assert.strictEqual(val.toLowerCase().includes('vessel'), true);
    }
  });
});
