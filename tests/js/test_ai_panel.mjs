import test from 'node:test';
import assert from 'node:assert/strict';
import AIView from '../../static/ai-panel.js';
import fs from 'node:fs';

function documentStub() {
  const elements = new Map();
  function element() {
    return { hidden: false, textContent: '', children: [], attributes: {},
      listeners: {}, checked: true,
      addEventListener(event, callback) { this.listeners[event] = callback; },
      setAttribute(k, v) { this.attributes[k] = v; },
      append(...children) { this.children.push(...children); },
      replaceChildren() { this.children = []; },
      set innerHTML(_) { throw Error('Untrusted text must never use innerHTML'); },
    };
  }
  return { getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
    createElement: element };
}

test('toggle transmits an explicit boolean; disabled makes no explain request', async () => {
  assert.deepEqual(AIView.askPayload('Q', 'tfidf', false), { question: 'Q', retriever: 'tfidf', ai_mode: false, defer_ai: true });
  const calls = [];
  await AIView.ask({ question: 'Q', retriever: 'tfidf', enabled: false,
    api: async (path, options) => { calls.push([path, JSON.parse(options.body)]); return { status: 'answered', ai: { status: 'disabled' } }; },
    onRetrieval() {}, onAI() {},
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0][1].ai_mode, false);
});

test('retrieval becomes visible before AI completes and survives provider failure', async () => {
  const events = [];
  const result = await AIView.ask({ question: 'Q', retriever: 'tfidf', enabled: true,
    api: async (path, options) => {
      if (path === '/api/ask') return { answer: 'Original source answer', request_id: 'r', ai: { status: 'pending' } };
      assert.deepEqual(JSON.parse(options.body), { request_id: 'r', ai_mode: true });
      assert.deepEqual(events, ['retrieval']);
      throw Error('provider details must not escape');
    }, onRetrieval(r) { events.push('retrieval'); assert.equal(r.answer, 'Original source answer'); },
    onAI(ai) { events.push('ai'); assert.equal(ai.status, 'unavailable'); },
  });
  assert.equal(result.answer, 'Original source answer');
  assert.deepEqual(events, ['retrieval', 'ai']);
});

test('general question without a dataset completes the AI request and renders a labelled answer', async () => {
  const doc = documentStub();
  const calls = [];
  const result = await AIView.ask({ question: 'What is light?', retriever: 'tfidf', enabled: true,
    api: async (path) => {
      calls.push(path);
      return path === '/api/ask'
        ? { reason: 'no_dataset', request_id: 'general', sources: [], ai: { status: 'pending' } }
        : { ai: { status: 'success', basis: 'general_knowledge', document_loaded: false,
          answer: 'Light is electromagnetic radiation.', latency_ms: 11800 } };
    }, onRetrieval() {}, onAI(ai) { AIView.render(ai, doc); },
  });
  assert.deepEqual(calls, ['/api/ask', '/api/explain']);
  assert.equal(result.ai.status, 'success');
  assert.equal(doc.getElementById('aiBasis').textContent, 'General AI knowledge · No dataset loaded');
  assert.equal(doc.getElementById('aiText').textContent, 'Light is electromagnetic radiation.');
  assert.equal(doc.getElementById('aiDetails').textContent, '11.8 s');
});

test('general AI completion failure never claims a retrieved answer exists', async () => {
  const result = await AIView.ask({ question: 'What is light?', retriever: 'tfidf', enabled: true,
    api: async (path) => {
      if (path === '/api/ask') return { reason: 'no_dataset', request_id: 'general', ai: { status: 'pending' } };
      throw Error('Offline');
    }, onRetrieval() {}, onAI() {},
  });
  assert.equal(result.ai.status, 'unavailable');
  assert.doesNotMatch(result.ai.message, /retrieved result/);
});

test('general knowledge has a clear label and malicious markup remains text', () => {
  const doc = documentStub();
  AIView.render({ status: 'success', basis: 'general_knowledge', answer: '<img src=x onerror=alert(1)>', model: 'Gemini' }, doc);
  assert.match(doc.getElementById('aiBasis').textContent, /Not verified/);
  assert.equal(doc.getElementById('aiText').textContent, '<img src=x onerror=alert(1)>');
  assert.equal(doc.getElementById('aiPanel').hidden, false);
});

test('AI explanation displays clean bullets without Markdown or passage tags, preserving the response', () => {
  const doc = documentStub();
  const ai = { status: 'success', basis: 'document_evidence', model: 'gemini-3.6-flash', latency_ms: 11800,
    context_truncated: true,
    answer: String.raw`## Key characteristics
Binarized Naïve Bayes clips counts to 1 [P20, P21].&#x20;
\* \*\*Performance:\*\* Binary features work well [P24].
* **Alternatives:** MaxEnt and SVMs [P24, P26].
- *Example:* The word fantastic appears once [P20].` };
  const original = structuredClone(ai);
  AIView.render(ai, doc);
  assert.equal(doc.getElementById('aiText').textContent, `Key characteristics
Binarized Naïve Bayes clips counts to 1.
• Performance: Binary features work well.
• Alternatives: MaxEnt and SVMs.
• Example: The word fantastic appears once.`);
  assert.equal(doc.getElementById('aiDetails').textContent, '11.8 s');
  assert.deepEqual(ai, original);
});

test('display cleanup preserves mathematical asterisks, ordinary brackets and safe literal HTML', () => {
  const doc = documentStub();
  const answer = 'Compute 2 * 3 * 4 = 24. Keep [x] and array[i]. <script>alert(1)</script>';
  AIView.render({ status: 'success', answer, model: 'hidden-model' }, doc);
  assert.equal(doc.getElementById('aiText').textContent, answer);
  assert.equal(doc.getElementById('aiDetails').textContent, '');
});

test('loading, disabled, unavailable and conflicting states are distinct', () => {
  const doc = documentStub();
  AIView.render({ status: 'pending' }, doc);
  assert.equal(doc.getElementById('aiPanel').attributes['aria-busy'], 'true');
  AIView.render({ status: 'success', basis: 'conflicting_evidence', answer: 'Conflicting values.' }, doc);
  assert.match(doc.getElementById('aiBasis').textContent, /No certain conclusion/);
  AIView.render({ status: 'unavailable', message: 'Try later.' }, doc);
  assert.equal(doc.getElementById('aiText').textContent, 'Try later.');
  AIView.render({ status: 'disabled' }, doc);
  assert.equal(doc.getElementById('aiPanel').hidden, true);
});

test('source panels use original text safely, including conflict alternatives', () => {
  const doc = documentStub();
  AIView.sources({ sources: [
    { passage_id: 1, passage_text: 'Pressure is 17 pascals <script>danger</script>.' },
    { passage_id: 2, passage_text: 'Pressure is 19 pascals.' },
  ] }, doc);
  const panels = doc.getElementById('retrievalSources').children;
  assert.equal(panels.length, 2);
  assert.match(panels[0].children[1].textContent, /<script>/);
});

test('actual page wires default-on toggle, distinct headings and module before script', () => {
  const html = fs.readFileSync(new URL('../../templates/index.html', import.meta.url), 'utf8');
  const script = fs.readFileSync(new URL('../../static/script.js', import.meta.url), 'utf8');
  assert.match(html, /id="aiMode"[^>]*checked/);
  assert.match(html, /Retrieved Text/);
  assert.match(html, /AI-Generated Explanation/);
  assert.ok(html.indexOf("filename='ai-panel.js'") < html.indexOf("filename='script.js'"));
  assert.match(script, /AIView.bindToggle\(document/);
  assert.match(script, /state.aiMode = enabled/);
  assert.match(script, /enabled: state.aiMode/);
});

test('native switch defaults and ON/OFF changes use the same submission state', () => {
  const doc = documentStub();
  const input = doc.getElementById('aiMode');
  const modes = [];
  AIView.bindToggle(doc, (enabled) => modes.push(enabled));
  assert.equal(doc.getElementById('aiModeLabel').textContent, 'ON');
  input.checked = false; input.listeners.change();
  assert.equal(doc.getElementById('aiModeLabel').textContent, 'OFF');
  input.checked = true; input.listeners.change();
  assert.deepEqual(modes, [true, false, true]);
  assert.equal(doc.getElementById('aiModeLabel').textContent, 'ON');
});

test('switch styles cover focus, disabled and reduced-motion states; helper is absent', () => {
  const css = fs.readFileSync(new URL('../../static/style.css', import.meta.url), 'utf8');
  const html = fs.readFileSync(new URL('../../templates/index.html', import.meta.url), 'utf8');
  const script = fs.readFileSync(new URL('../../static/script.js', import.meta.url), 'utf8');
  assert.match(css, /\.ai-switch input:focus-visible \+ \.ai-switch-track/);
  assert.match(css, /\.ai-switch input:checked \+ \.ai-switch-track/);
  assert.match(css, /\.ai-switch input:disabled/);
  assert.match(css, /prefers-reduced-motion: reduce/);
  assert.match(css, /translateX\(20px\)/);
  assert.ok(!html.includes('When on, your question and selected evidence are sent to Google Gemini.'));
  assert.ok(!html.includes('aiConfigStatus') && !script.includes('aiConfigStatus'));
});
