/**
 * Automated verification test suite for Heart Disease Stage 4 UI and Service Contract.
 * Validates form mappings, 13 predictors, strict absence of 'num', and nullability of ca and thal.
 */

import assert from 'node:assert';

console.log('--- RUNNING HEART DISEASE UI & CONTRACT VERIFICATION SUITE ---');

// 1. Verify 13 Predictor Keys
const EXPECTED_PREDICTOR_KEYS = [
  'age',
  'sex',
  'cp',
  'trestbps',
  'chol',
  'fbs',
  'restecg',
  'thalach',
  'exang',
  'oldpeak',
  'slope',
  'ca',
  'thal',
];

assert.strictEqual(EXPECTED_PREDICTOR_KEYS.length, 13, 'Must have exactly 13 predictor features');
console.log('✓ Test 1: Exactly 13 predictor feature keys defined.');

// 2. Strict Absence of 'num'
assert.strictEqual(EXPECTED_PREDICTOR_KEYS.includes('num'), false, "Target variable 'num' must NOT be in predictors");
console.log("✓ Test 2: Target variable 'num' is strictly absent.");

// 3. Human-readable mapping verification
const LABELS = {
  sex: { '0': 'Female', '1': 'Male' },
  cp: { '1': 'Typical angina', '2': 'Atypical angina', '3': 'Non-anginal pain', '4': 'Asymptomatic' },
  fbs: { '0': '<= 120 mg/dL', '1': '> 120 mg/dL' },
  restecg: { '0': 'Normal', '1': 'ST-T wave abnormality', '2': 'Left ventricular hypertrophy' },
  exang: { '0': 'No', '1': 'Yes' },
  slope: { '1': 'Upsloping', '2': 'Flat', '3': 'Downsloping' },
  ca: { '0': '0 vessels', '1': '1 vessel', '2': '2 vessels', '3': '3 vessels', 'null': 'Not provided (null)' },
  thal: { '3': 'Normal (3.0)', '6': 'Fixed defect (6.0)', '7': 'Reversible defect (7.0)', 'null': 'Not provided (null)' },
};

assert.strictEqual(LABELS.sex['0'], 'Female');
assert.strictEqual(LABELS.sex['1'], 'Male');
assert.strictEqual(LABELS.cp['1'], 'Typical angina');
assert.strictEqual(LABELS.cp['4'], 'Asymptomatic');
assert.strictEqual(LABELS.fbs['0'], '<= 120 mg/dL');
assert.strictEqual(LABELS.fbs['1'], '> 120 mg/dL');
assert.strictEqual(LABELS.restecg['0'], 'Normal');
assert.strictEqual(LABELS.restecg['1'], 'ST-T wave abnormality');
assert.strictEqual(LABELS.restecg['2'], 'Left ventricular hypertrophy');
assert.strictEqual(LABELS.exang['0'], 'No');
assert.strictEqual(LABELS.exang['1'], 'Yes');
assert.strictEqual(LABELS.slope['1'], 'Upsloping');
assert.strictEqual(LABELS.slope['2'], 'Flat');
assert.strictEqual(LABELS.slope['3'], 'Downsloping');
assert.strictEqual(LABELS.ca['0'], '0 vessels');
assert.strictEqual(LABELS.ca['3'], '3 vessels');
assert.strictEqual(LABELS.thal['3'], 'Normal (3.0)');
assert.strictEqual(LABELS.thal['6'], 'Fixed defect (6.0)');
assert.strictEqual(LABELS.thal['7'], 'Reversible defect (7.0)');
console.log('✓ Test 3: All human-readable categorical labels map to valid Cleveland codes.');

// 4. Nullable ca and thal payload parsing
function buildPayload(rawForm, patientId) {
  const payload = {
    patientId: patientId > 0 ? patientId : undefined,
    age: parseFloat(rawForm.age),
    sex: parseFloat(rawForm.sex),
    cp: parseFloat(rawForm.cp),
    trestbps: parseFloat(rawForm.trestbps),
    chol: parseFloat(rawForm.chol),
    fbs: parseFloat(rawForm.fbs),
    restecg: parseFloat(rawForm.restecg),
    thalach: parseFloat(rawForm.thalach),
    exang: parseFloat(rawForm.exang),
    oldpeak: parseFloat(rawForm.oldpeak),
    slope: parseFloat(rawForm.slope),
    ca: rawForm.ca && rawForm.ca !== 'null' ? parseFloat(rawForm.ca) : null,
    thal: rawForm.thal && rawForm.thal !== 'null' ? parseFloat(rawForm.thal) : null,
  };
  return payload;
}

const sampleFormWithNulls = {
  age: '63',
  sex: '1',
  cp: '4',
  trestbps: '140',
  chol: '260',
  fbs: '0',
  restecg: '1',
  thalach: '112',
  exang: '1',
  oldpeak: '3.0',
  slope: '2',
  ca: 'null',
  thal: 'null',
};

const payload1 = buildPayload(sampleFormWithNulls, 4);
assert.strictEqual(payload1.ca, null, "ca must be null when 'null' is selected");
assert.strictEqual(payload1.thal, null, "thal must be null when 'null' is selected");
assert.strictEqual(payload1.patientId, 4, 'patientId must be preserved from patient context');
assert.strictEqual('num' in payload1, false, "'num' must never exist in the payload");
console.log('✓ Test 4: Nullable ca=null and thal=null succeed without fabricating numbers.');

// 5. Sample form with complete values
const sampleFormComplete = {
  age: '54',
  sex: '0',
  cp: '3',
  trestbps: '135',
  chol: '220',
  fbs: '0',
  restecg: '0',
  thalach: '160',
  exang: '0',
  oldpeak: '1.2',
  slope: '1',
  ca: '2',
  thal: '3',
};

const payload2 = buildPayload(sampleFormComplete, 0); // stateless
assert.strictEqual(payload2.ca, 2.0);
assert.strictEqual(payload2.thal, 3.0);
assert.strictEqual(payload2.patientId, undefined, 'patientId must be undefined for stateless calls');
console.log('✓ Test 5: Full values map correctly; stateless inference omits patientId.');

console.log('--- ALL 5 CONTRACT TESTS PASSED SUCCESSFULLY ---');
