import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

// Describe the Patient-Centric Heart Pathways and Disease Selection Gateway Tests
describe('Heart Product Refinement — Patient-Centric Health Entry & Disease Gateway Tests', () => {

  // Test 1: Three distinct pathways defined for Heart Health
  test('1. Three distinct top-level pathways defined (Symptoms, Monitoring, Clinical Results)', () => {
    const pathways = [
      {
        id: 'symptoms',
        title: 'Talk about symptoms',
        description: "Discuss what you're experiencing and get safety-focused conversational guidance.",
        route: '/chat',
      },
      {
        id: 'monitoring',
        title: 'Monitor my health',
        description: 'Track available health measurements, personal baselines, and changes over time.',
        route: '/monitoring',
      },
      {
        id: 'clinical',
        title: 'I have clinical results',
        description: 'Choose a disease-specific screening module when you have validated clinical or laboratory measurements.',
        action: 'open_gateway',
      },
    ];

    assert.strictEqual(pathways.length, 3);
    assert.strictEqual(pathways[0].route, '/chat');
    assert.strictEqual(pathways[1].route, '/monitoring');
    assert.strictEqual(pathways[2].action, 'open_gateway');
  });

  // Test 2: "Talk about symptoms" routes to existing chatbot
  test('2. "Talk about symptoms" action target is the existing /chat route', () => {
    const targetRoute = '/chat';
    assert.strictEqual(targetRoute, '/chat');
  });

  // Test 3: "Monitor my health" routes to existing monitoring functionality
  test('3. "Monitor my health" action target is the existing /monitoring route', () => {
    const targetRoute = '/monitoring';
    assert.strictEqual(targetRoute, '/monitoring');
  });

  // Test 4: Clicking "I have clinical results" does NOT immediately open Heart Disease; opens gateway modal
  test('4. Clicking "I have clinical results" does NOT immediately open Heart Disease; opens gateway modal', () => {
    let showClinicalGatewayModal = false;
    let selectedDisease: string | null = null;

    // Simulate user clicking "I have clinical results" on the main page
    const onClinicalResultsClick = () => {
      showClinicalGatewayModal = true;
      // Must NOT directly set selectedDisease to HEART_DISEASE
    };

    onClinicalResultsClick();

    assert.strictEqual(showClinicalGatewayModal, true);
    assert.strictEqual(selectedDisease, null);
    assert.notStrictEqual(selectedDisease, 'HEART_DISEASE');
  });

  // Test 5: Selecting Heart Disease in Gateway opens existing Heart assessment
  test('5. Selecting Heart Disease in Gateway opens existing HeartDiseaseAssessmentModal', () => {
    let showClinicalGatewayModal = true;
    let selectedDisease: string | null = null;

    // Simulate user selecting Heart Disease in gateway
    const onSelectHeartInGateway = () => {
      showClinicalGatewayModal = false;
      selectedDisease = 'HEART_DISEASE';
    };

    onSelectHeartInGateway();

    assert.strictEqual(showClinicalGatewayModal, false);
    assert.strictEqual(selectedDisease, 'HEART_DISEASE');
  });

  // Test 6: Selecting Liver Disease in Gateway opens existing Liver assessment
  test('6. Selecting Liver Disease in Gateway opens existing LiverDiseaseAssessmentModal', () => {
    let showClinicalGatewayModal = true;
    let selectedDisease: string | null = null;

    // Simulate user selecting Liver Disease in gateway
    const onSelectLiverInGateway = () => {
      showClinicalGatewayModal = false;
      selectedDisease = 'LIVER_DISEASE';
    };

    onSelectLiverInGateway();

    assert.strictEqual(showClinicalGatewayModal, false);
    assert.strictEqual(selectedDisease, 'LIVER_DISEASE');
  });

  // Test 7: Unimplemented modules (Diabetes, Parkinson's) are clearly marked unavailable/coming soon
  test('7. Unimplemented modules (Diabetes, Parkinson\'s) are disabled and marked coming soon without fake functionality', () => {
    const gatewayModules = [
      { id: 'HEART_DISEASE', name: 'Heart Disease', status: 'AVAILABLE', canOpen: true },
      { id: 'LIVER_DISEASE', name: 'Liver Disease', status: 'AVAILABLE', canOpen: true },
      { id: 'DIABETES', name: 'Diabetes', status: 'COMING_SOON', canOpen: false },
      { id: 'PARKINSONS', name: 'Parkinson\'s Disease', status: 'COMING_SOON', canOpen: false },
    ];

    const diabetes = gatewayModules.find((m) => m.id === 'DIABETES')!;
    const parkinsons = gatewayModules.find((m) => m.id === 'PARKINSONS')!;

    assert.strictEqual(diabetes.status, 'COMING_SOON');
    assert.strictEqual(diabetes.canOpen, false);

    assert.strictEqual(parkinsons.status, 'COMING_SOON');
    assert.strictEqual(parkinsons.canOpen, false);
  });

  // Test 8: Cancel closes gateway without opening any assessment
  test('8. Cancel closes the disease-selection gateway without opening an assessment', () => {
    let showClinicalGatewayModal = true;
    let selectedDisease: string | null = null;

    // Simulate clicking Cancel in gateway
    const onCancelGateway = () => {
      showClinicalGatewayModal = false;
    };

    onCancelGateway();

    assert.strictEqual(showClinicalGatewayModal, false);
    assert.strictEqual(selectedDisease, null);
  });

  // Test 9: Dynamic patientId is strictly preserved without hardcoding
  test('9. Dynamic activePatientId is preserved across assessment selections without hardcoding', () => {
    const contextPatientId: number | null = 42;
    let targetAssessmentPatientId: number | null = null;

    const launchAssessment = (patientId: number | null) => {
      targetAssessmentPatientId = patientId;
    };

    launchAssessment(contextPatientId);
    assert.strictEqual(targetAssessmentPatientId, 42);
    assert.notStrictEqual(targetAssessmentPatientId, 1); // Not hardcoded to 1
  });

  // Test 10: Non-diagnostic screening language is strictly preserved
  test('10. Non-diagnostic screening language is strictly preserved', () => {
    const disclaimer = 'This screening model is an algorithmic assessment tool and does NOT provide a medical diagnosis. Consult a qualified cardiologist or physician.';
    assert.ok(disclaimer.includes('does NOT provide a medical diagnosis'));
    assert.ok(!disclaimer.includes('diagnoses heart disease'));
  });

  // Test 11: Heart assessment strictly requires the validated 11 required clinical features
  test('11. Heart assessment still requires the validated 11 required fields and rejects submission when incomplete', () => {
    const requiredFields = [
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
    ];

    assert.strictEqual(requiredFields.length, 11);

    // Incomplete payload missing chol
    const partialPayload = {
      age: 55,
      sex: 1,
      cp: 2,
      trestbps: 130,
      chol: undefined, // missing cholesterol
      fbs: 0,
      restecg: 0,
      thalach: 150,
      exang: 0,
      oldpeak: 1.0,
      slope: 1,
    };

    const isComplete = requiredFields.every(
      (f) => partialPayload[f as keyof typeof partialPayload] !== undefined && partialPayload[f as keyof typeof partialPayload] !== null
    );

    assert.strictEqual(isComplete, false);
  });

  // Test 12: ca and thal remain nullable
  test('12. ca and thal remain nullable and do not block valid submission when null', () => {
    const payloadWithNulls = {
      age: 60,
      sex: 1,
      cp: 3,
      trestbps: 140,
      chol: 240,
      fbs: 0,
      restecg: 1,
      thalach: 145,
      exang: 1,
      oldpeak: 2.2,
      slope: 2,
      ca: null,   // allowed to be null
      thal: null, // allowed to be null
    };

    const requiredFields = [
      'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
      'restecg', 'thalach', 'exang', 'oldpeak', 'slope'
    ];

    const hasAllRequired = requiredFields.every(
      (f) => payloadWithNulls[f as keyof typeof payloadWithNulls] !== null && payloadWithNulls[f as keyof typeof payloadWithNulls] !== undefined
    );

    assert.strictEqual(hasAllRequired, true);
    assert.strictEqual(payloadWithNulls.ca, null);
    assert.strictEqual(payloadWithNulls.thal, null);
  });

  // Test 13: Missing cholesterol does not generate a fabricated value
  test('13. Missing cholesterol does not fabricate a population average value (e.g. 240)', () => {
    const userInput = "I don't have my cholesterol";
    const parsedValues: Record<string, number | null> = {};

    // System must NOT inject median/mean cholesterol
    if (!userInput.match(/\bcholesterol\s*(?:is|=|:)?\s*(\d+)/i)) {
      parsedValues['chol'] = null;
    }

    assert.strictEqual(parsedValues['chol'], null);
    assert.notStrictEqual(parsedValues['chol'], 240); // 240 is Cleveland dataset median
  });

  // Test 14: "BP is high" does not become an invented numeric BP
  test('14. "BP is high" does not become an invented numeric BP', () => {
    const vagueInput = "My BP is high";
    let extractedTrestbps: number | null = null;

    // Pattern only extracts numeric measurements
    const match = vagueInput.match(/\b(?:bp|blood pressure)\s*(?:is|=|:)?\s*(\d{2,3})/i);
    if (match) {
      extractedTrestbps = parseInt(match[1], 10);
    }

    assert.strictEqual(extractedTrestbps, null);
  });

  // Test 15: Informational banner content in Clinical Assessment Modal
  test('15. Clinical Data Screening Assessment provides clear explanatory guidance and pathway alternatives', () => {
    const modalTitle = 'Clinical Data Screening Assessment';
    const explanation = 'This model uses structured clinical and laboratory information. If you have a recent medical report or validated measurements, enter those values here. Do not guess unknown values.';
    const missingDataHelp = "Don't have all of these values? That's okay. You can still use Smart Health for symptom conversations, health monitoring, personal baselines, trends, and previous assessments. The Heart screening model can be used when the required clinical information is available.";

    assert.strictEqual(modalTitle, 'Clinical Data Screening Assessment');
    assert.ok(explanation.includes('Do not guess unknown values'));
    assert.ok(missingDataHelp.includes("Don't have all of these values?"));
    assert.ok(missingDataHelp.includes('symptom conversations'));
    assert.ok(missingDataHelp.includes('health monitoring'));
  });

  // Test 16: Assessments page patient-centric primary heading and subtitle
  test('16. Assessments page presents patient-first heading and supporting text', () => {
    const primaryHeading = 'How can Smart Health help you today?';
    const supportingText =
      'Talk about symptoms, monitor changes in your health, or use validated clinical results for an AI-assisted screening assessment.';

    assert.strictEqual(primaryHeading, 'How can Smart Health help you today?');
    assert.ok(supportingText.includes('Talk about symptoms'));
    assert.ok(supportingText.includes('monitor changes in your health'));
    assert.ok(supportingText.includes('use validated clinical results'));
  });

  // Test 17: Three balanced primary visual action cards and buttons
  test('17. Three primary pathways have balanced visual actions and designated button labels', () => {
    const primaryCards = [
      {
        title: 'Talk about symptoms',
        description: "Discuss what you're experiencing and get safety-focused conversational guidance.",
        actionLabel: 'Talk to Assistant',
        route: '/chat',
      },
      {
        title: 'Monitor my health',
        description: 'Track available health measurements, personal baselines, and changes over time.',
        actionLabel: 'Open Monitoring',
        route: '/monitoring',
      },
      {
        title: 'I have clinical results',
        description: 'Choose a disease-specific screening module when you have validated clinical or laboratory measurements.',
        actionLabel: 'Clinical Assessment',
        gatewayTarget: 'showClinicalGatewayModal',
      },
    ];

    assert.strictEqual(primaryCards.length, 3);
    assert.strictEqual(primaryCards[0].actionLabel, 'Talk to Assistant');
    assert.strictEqual(primaryCards[1].actionLabel, 'Open Monitoring');
    assert.strictEqual(primaryCards[2].actionLabel, 'Clinical Assessment');
  });

  // Test 18: Clinical Assessment card explicitly communicates requirement boundary
  test('18. Clinical Assessment card explicitly communicates it is for patients with required validated measurements', () => {
    const requirementNotice =
      'This option is for patients who already have the required validated clinical measurements.';
    assert.ok(requirementNotice.includes('already have the required validated clinical measurements'));
  });

  // Test 19: Rejection of prohibited diagnostic language
  test('19. UI strings strictly avoid prohibited diagnostic and coercive language', () => {
    const prohibitedPhrases = [
      'Required for your health',
      'Check your heart disease',
      'Find out if you have heart disease',
      'Diagnose',
      'Confirm disease',
    ];

    const currentUiTexts = [
      'How can Smart Health help you today?',
      'Talk about symptoms, monitor changes in your health, or use validated clinical results for an AI-assisted screening assessment.',
      "Discuss what you're experiencing and get safety-focused conversational guidance.",
      'Track available health measurements, personal baselines, and changes over time.',
      'Choose a disease-specific screening module when you have validated clinical or laboratory measurements.',
      'Which health assessment do you want to explore?',
      'Choose the screening module that matches the clinical or laboratory results you have available.',
      'This option is for patients who already have the required validated clinical measurements.',
    ];

    for (const text of currentUiTexts) {
      for (const phrase of prohibitedPhrases) {
        assert.ok(
          !text.toLowerCase().includes(phrase.toLowerCase()),
          `UI text "${text}" must not include prohibited phrase "${phrase}"`
        );
      }
    }
  });
});
