import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  resolveOneHotFeatureStatus,
  parseHeartAssessment,
} from '../src/utils/heartAssessmentFormatter.ts';
import type { DiseaseAssessment } from '../src/types/assessment.ts';

describe('Stage 8 Final Closure — GAP 1: SHAP Transformed-Feature to UI Mapping', () => {
  // ============================================================
  // Direct Categorical Group Cases A through G
  // ============================================================

  test('CASE A: cp=3 -> verify cp_3 is active and cp_4 is inactive', () => {
    const rawCp = 3;
    const cp3Status = resolveOneHotFeatureStatus('cat__cp_3.0', rawCp);
    const cp4Status = resolveOneHotFeatureStatus('cat__cp_4.0', rawCp);

    assert.strictEqual(cp3Status.isCategorical, true);
    assert.strictEqual(cp3Status.isActive, true);
    assert.strictEqual(cp3Status.transformedValue, 1);
    assert.strictEqual(cp3Status.displayStatus, 'Active (1)');

    assert.strictEqual(cp4Status.isCategorical, true);
    assert.strictEqual(cp4Status.isActive, false);
    assert.strictEqual(cp4Status.transformedValue, 0);
    assert.strictEqual(cp4Status.displayStatus, 'Inactive (0)');
  });

  test('CASE B: cp=4 -> verify cp_4 is active and cp_3 is inactive', () => {
    const rawCp = 4;
    const cp4Status = resolveOneHotFeatureStatus('cat__cp_4.0', rawCp);
    const cp3Status = resolveOneHotFeatureStatus('cat__cp_3.0', rawCp);

    assert.strictEqual(cp4Status.isCategorical, true);
    assert.strictEqual(cp4Status.isActive, true);
    assert.strictEqual(cp4Status.transformedValue, 1);
    assert.strictEqual(cp4Status.displayStatus, 'Active (1)');

    assert.strictEqual(cp3Status.isCategorical, true);
    assert.strictEqual(cp3Status.isActive, false);
    assert.strictEqual(cp3Status.transformedValue, 0);
    assert.strictEqual(cp3Status.displayStatus, 'Inactive (0)');
  });

  test('CASE C: thal=3 -> verify thal_6 and thal_7 are inactive (reference category)', () => {
    const rawThal = 3;
    const thal6Status = resolveOneHotFeatureStatus('cat__thal_6.0', rawThal);
    const thal7Status = resolveOneHotFeatureStatus('cat__thal_7.0', rawThal);

    assert.strictEqual(thal6Status.isCategorical, true);
    assert.strictEqual(thal6Status.isActive, false);
    assert.strictEqual(thal6Status.transformedValue, 0);
    assert.strictEqual(thal6Status.displayStatus, 'Inactive (0)');

    assert.strictEqual(thal7Status.isCategorical, true);
    assert.strictEqual(thal7Status.isActive, false);
    assert.strictEqual(thal7Status.transformedValue, 0);
    assert.strictEqual(thal7Status.displayStatus, 'Inactive (0)');
  });

  test('CASE D: thal=7 -> verify thal_7 is active and thal_6 is inactive', () => {
    const rawThal = 7;
    const thal7Status = resolveOneHotFeatureStatus('cat__thal_7.0', rawThal);
    const thal6Status = resolveOneHotFeatureStatus('cat__thal_6.0', rawThal);

    assert.strictEqual(thal7Status.isCategorical, true);
    assert.strictEqual(thal7Status.isActive, true);
    assert.strictEqual(thal7Status.transformedValue, 1);
    assert.strictEqual(thal7Status.displayStatus, 'Active (1)');

    assert.strictEqual(thal6Status.isCategorical, true);
    assert.strictEqual(thal6Status.isActive, false);
    assert.strictEqual(thal6Status.transformedValue, 0);
    assert.strictEqual(thal6Status.displayStatus, 'Inactive (0)');
  });

  test('CASE E: slope=1 -> verify slope_2 and slope_3 are inactive (reference category)', () => {
    const rawSlope = 1;
    const slope2Status = resolveOneHotFeatureStatus('cat__slope_2.0', rawSlope);
    const slope3Status = resolveOneHotFeatureStatus('cat__slope_3.0', rawSlope);

    assert.strictEqual(slope2Status.isCategorical, true);
    assert.strictEqual(slope2Status.isActive, false);
    assert.strictEqual(slope2Status.transformedValue, 0);
    assert.strictEqual(slope2Status.displayStatus, 'Inactive (0)');

    assert.strictEqual(slope3Status.isCategorical, true);
    assert.strictEqual(slope3Status.isActive, false);
    assert.strictEqual(slope3Status.transformedValue, 0);
    assert.strictEqual(slope3Status.displayStatus, 'Inactive (0)');
  });

  test('CASE F: slope=2 -> verify slope_2 is active and slope_3 is inactive', () => {
    const rawSlope = 2;
    const slope2Status = resolveOneHotFeatureStatus('cat__slope_2.0', rawSlope);
    const slope3Status = resolveOneHotFeatureStatus('cat__slope_3.0', rawSlope);

    assert.strictEqual(slope2Status.isCategorical, true);
    assert.strictEqual(slope2Status.isActive, true);
    assert.strictEqual(slope2Status.transformedValue, 1);
    assert.strictEqual(slope2Status.displayStatus, 'Active (1)');

    assert.strictEqual(slope3Status.isCategorical, true);
    assert.strictEqual(slope3Status.isActive, false);
    assert.strictEqual(slope3Status.transformedValue, 0);
    assert.strictEqual(slope3Status.displayStatus, 'Inactive (0)');
  });

  test('CASE G: restecg=2 -> verify restecg_2 is active and restecg_1 is inactive', () => {
    const rawRestecg = 2;
    const restecg2Status = resolveOneHotFeatureStatus('cat__restecg_2.0', rawRestecg);
    const restecg1Status = resolveOneHotFeatureStatus('cat__restecg_1.0', rawRestecg);

    assert.strictEqual(restecg2Status.isCategorical, true);
    assert.strictEqual(restecg2Status.isActive, true);
    assert.strictEqual(restecg2Status.transformedValue, 1);
    assert.strictEqual(restecg2Status.displayStatus, 'Active (1)');

    assert.strictEqual(restecg1Status.isCategorical, true);
    assert.strictEqual(restecg1Status.isActive, false);
    assert.strictEqual(restecg1Status.transformedValue, 0);
    assert.strictEqual(restecg1Status.displayStatus, 'Inactive (0)');
  });

  // ============================================================
  // Complete End-to-End parseHeartAssessment Verification
  // ============================================================

  test('UI Mapping: parseHeartAssessment maps one-hot features and preserves exact SHAP values', () => {
    const assessment: DiseaseAssessment = {
      id: 7001,
      patientId: 88,
      diseaseType: 'HEART_DISEASE',
      riskScore: 0.28,
      riskLevel: 'LOW',
      assessedAt: '2026-10-02T12:00:00Z',
      inputFeatures: JSON.stringify({
        probability: 0.28,
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
        shapBaseValue: -0.0879,
        modelOutput: -0.944,
        inputSummary: {
          age: 55,
          sex: 1,
          cp: 3, // Non-Anginal
          trestbps: 130,
          chol: 230,
          fbs: 0,
          restecg: 2, // LVH
          thalach: 155,
          exang: 0,
          oldpeak: 1.0,
          slope: 2, // Flat
          ca: null,
          thal: 7, // Reversible Defect
        },
        positiveContributors: [
          {
            feature: 'cp',
            transformedFeature: 'cat__cp_3.0',
            displayName: 'Chest Pain: Non-Anginal',
            rawValue: 3,
            contribution: 0.4521,
            absContribution: 0.4521,
            direction: 'INCREASES_RISK_ESTIMATE',
          },
          {
            feature: 'thal',
            transformedFeature: 'cat__thal_7.0',
            displayName: 'Thallium: Reversible Defect',
            rawValue: 7,
            contribution: 0.3812,
            absContribution: 0.3812,
            direction: 'INCREASES_RISK_ESTIMATE',
          },
        ],
        negativeContributors: [
          {
            feature: 'cp',
            transformedFeature: 'cat__cp_4.0',
            displayName: 'Chest Pain: Asymptomatic',
            rawValue: 3,
            contribution: -0.3105,
            absContribution: 0.3105,
            direction: 'DECREASES_RISK_ESTIMATE',
          },
          {
            feature: 'thal',
            transformedFeature: 'cat__thal_6.0',
            displayName: 'Thallium: Fixed Defect',
            rawValue: 7,
            contribution: -0.1984,
            absContribution: 0.1984,
            direction: 'DECREASES_RISK_ESTIMATE',
          },
        ],
        allContributions: [],
      }),
    };

    const parsed = parseHeartAssessment(assessment);

    // Verify positive contributors
    const posCp3 = parsed.positiveContributors.find((c) => c.transformedFeature === 'cat__cp_3.0');
    assert.ok(posCp3);
    assert.strictEqual(posCp3.isCategorical, true);
    assert.strictEqual(posCp3.isActive, true);
    assert.strictEqual(posCp3.transformedValue, 1);
    assert.strictEqual(posCp3.displayStatus, 'Active (1)');
    // Crucial: SHAP attribution value is strictly unchanged
    assert.strictEqual(posCp3.contribution, 0.4521);

    const posThal7 = parsed.positiveContributors.find((c) => c.transformedFeature === 'cat__thal_7.0');
    assert.ok(posThal7);
    assert.strictEqual(posThal7.isCategorical, true);
    assert.strictEqual(posThal7.isActive, true);
    assert.strictEqual(posThal7.transformedValue, 1);
    assert.strictEqual(posThal7.displayStatus, 'Active (1)');
    assert.strictEqual(posThal7.contribution, 0.3812);

    // Verify negative contributors (inactive one-hot categories)
    const negCp4 = parsed.negativeContributors.find((c) => c.transformedFeature === 'cat__cp_4.0');
    assert.ok(negCp4);
    assert.strictEqual(negCp4.isCategorical, true);
    assert.strictEqual(negCp4.isActive, false);
    assert.strictEqual(negCp4.transformedValue, 0);
    assert.strictEqual(negCp4.displayStatus, 'Inactive (0)');
    // Crucial: SHAP attribution value is strictly unchanged
    assert.strictEqual(negCp4.contribution, -0.3105);

    const negThal6 = parsed.negativeContributors.find((c) => c.transformedFeature === 'cat__thal_6.0');
    assert.ok(negThal6);
    assert.strictEqual(negThal6.isCategorical, true);
    assert.strictEqual(negThal6.isActive, false);
    assert.strictEqual(negThal6.transformedValue, 0);
    assert.strictEqual(negThal6.displayStatus, 'Inactive (0)');
    assert.strictEqual(negThal6.contribution, -0.1984);
  });
});
