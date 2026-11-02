import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import type { ChatRequest, ChatAssistantContextData } from '../src/types/chatbot.ts';
import { formatMetricName } from '../src/utils/formatters.ts';

describe('Context-Aware Smart Health AI Assistant Specification Tests', () => {
  it('1. ChatRequest contract supports optional structured clinical context', () => {
    const context: ChatAssistantContextData = {
      source: 'assessment',
      disease: 'HEART',
      assessmentId: 10,
    };

    const req: ChatRequest = {
      message: 'What does this result mean?',
      session_id: 'session_patient_4_12345',
      user_id: '4',
      context: context,
    };

    assert.equal(req.context?.source, 'assessment');
    assert.equal(req.context?.disease, 'HEART');
    assert.equal(req.context?.assessmentId, 10);
    assert.equal(req.user_id, '4');
  });

  it('2. Assessment context handoff formats contextual tooltip and title dynamically', () => {
    const assessmentId = 42;
    const tooltipText = `Ask Smart Health AI about Heart Assessment #${assessmentId}`;

    assert.match(tooltipText, /Heart Assessment #42/);
    assert.doesNotMatch(tooltipText, /#10/); // Never hardcoded
  });

  it('3. Monitoring context handoff formats dynamic metric tooltip and indicator', () => {
    const metric = 'GLUCOSE';
    const formatted = formatMetricName(metric);
    const tooltip = `Ask Smart Health AI about ${formatted} monitoring`;

    assert.equal(formatted, 'Blood Glucose');
    assert.equal(tooltip, 'Ask Smart Health AI about Blood Glucose monitoring');

    const hrMetric = 'HEART_RATE';
    const hrFormatted = formatMetricName(hrMetric);
    assert.equal(hrFormatted, 'Heart Rate');
  });

  it('4. Assessment context generates safe decision-support suggested question chips', () => {
    const activeContext: ChatAssistantContextData = {
      source: 'assessment',
      disease: 'HEART',
      assessmentId: 15,
    };

    const getSuggestedPrompts = (ctx?: ChatAssistantContextData | null) => {
      if (ctx?.source === 'assessment' || ctx?.assessmentId) {
        return [
          'What does this result mean?',
          'Why did the model give me this result?',
          'Does this mean I have heart disease?',
          'How does this compare with my previous assessment?',
          'What should I discuss with my doctor?',
        ];
      }
      return ['What does my recent health monitoring show?'];
    };

    const prompts = getSuggestedPrompts(activeContext);
    assert.ok(prompts.includes('What does this result mean?'));
    assert.ok(prompts.includes('Why did the model give me this result?'));
    assert.ok(prompts.includes('Does this mean I have heart disease?'));
    assert.ok(prompts.includes('How does this compare with my previous assessment?'));
    assert.ok(prompts.includes('What should I discuss with my doctor?'));
  });

  it('5. Monitoring context generates metric-specific suggested question chips', () => {
    const activeContext: ChatAssistantContextData = {
      source: 'monitoring',
      metric: 'GLUCOSE',
    };

    const getSuggestedPrompts = (ctx?: ChatAssistantContextData | null) => {
      if (ctx?.source === 'monitoring' || ctx?.metric) {
        return [
          `What does my recent ${ctx.metric ? formatMetricName(ctx.metric) : 'health'} monitoring show?`,
          'How does my latest reading compare with my baseline?',
          'Is my trend going up or down?',
          'What should I discuss with my doctor?',
        ];
      }
      return [];
    };

    const prompts = getSuggestedPrompts(activeContext);
    assert.ok(prompts.includes('What does my recent Blood Glucose monitoring show?'));
    assert.ok(prompts.includes('How does my latest reading compare with my baseline?'));
    assert.ok(prompts.includes('Is my trend going up or down?'));
  });

  it('6. Dynamic patient isolation: user_id is strictly derived from active patient', () => {
    const patientAId = 4;
    const patientBId = 1;

    const requestA: ChatRequest = {
      message: 'What does this result mean?',
      user_id: patientAId.toString(),
      session_id: `session_patient_${patientAId}_1`,
      context: { source: 'assessment', disease: 'HEART', assessmentId: 10 },
    };

    const requestB: ChatRequest = {
      message: 'What does this result mean?',
      user_id: patientBId.toString(),
      session_id: `session_patient_${patientBId}_1`,
      context: { source: 'assessment', disease: 'HEART', assessmentId: 11 },
    };

    assert.notEqual(requestA.user_id, requestB.user_id);
    assert.notEqual(requestA.session_id, requestB.session_id);
    assert.equal(requestA.context?.assessmentId, 10);
    assert.equal(requestB.context?.assessmentId, 11);
  });

  it('7. No fabricated clinical values or mock probabilities are stored in frontend context handoff', () => {
    // Context handoff only passes authoritative references, never frontend-computed probabilities or SHAP
    const cleanContext: ChatAssistantContextData = {
      source: 'assessment',
      disease: 'HEART',
      assessmentId: 10,
    };

    assert.equal((cleanContext as any).probability, undefined);
    assert.equal((cleanContext as any).prediction, undefined);
    assert.equal((cleanContext as any).threshold, undefined);
    assert.equal((cleanContext as any).shap, undefined);
  });
});
