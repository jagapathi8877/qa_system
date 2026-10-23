/* Separate AI request and safe text rendering, shared with the Node tests. */
(function (root) {
  'use strict';
  const labels = {
    document_evidence: 'Based on selected document evidence · AI-generated',
    general_knowledge: 'General AI knowledge · Not verified in the uploaded document',
    conflicting_evidence: 'Conflicting document evidence · No certain conclusion',
    incomplete_evidence: 'Ambiguous or incomplete evidence · Clarification may be needed',
  };

  function askPayload(question, retriever, enabled) {
    return { question, retriever, ai_mode: enabled === true, defer_ai: true };
  }

  function bindToggle(document, onChange) {
    const input = document.getElementById('aiMode');
    const sync = () => {
      document.getElementById('aiModeLabel').textContent = input.checked ? 'ON' : 'OFF';
      onChange(input.checked);
    };
    input.addEventListener('change', sync);
    sync();
  }

  async function ask({ question, retriever, enabled, api, onRetrieval, onAI }) {
    const result = await api('/api/ask', {
      method: 'POST', body: JSON.stringify(askPayload(question, retriever, enabled)),
    });
    onRetrieval(result);
    if (result.ai && result.ai.status === 'pending') {
      try {
        const completed = await api('/api/explain', {
          method: 'POST', body: JSON.stringify({ request_id: result.request_id, ai_mode: enabled === true }),
        });
        result.ai = completed.ai;
      } catch (_) {
        result.ai = { enabled: true, status: 'unavailable', basis: 'unavailable',
          message: result.reason === 'no_dataset'
            ? 'The AI answer could not be completed. Please try again.'
            : 'The AI explanation could not be completed. The retrieved result is still available.' };
      }
    }
    onAI(result.ai);
    return result;
  }

  // Keep provider text and citations intact in the response; simplify only the display.
  // This remains plain text so model output can never create HTML elements.
  function explanationText(answer) {
    return String(answer || '')
      .replace(/&#(?:x20|32);|&nbsp;/gi, ' ')
      .replace(/\\([*_[\]`])/g, '$1')
      .replace(/[ \t]*\[P\d+(?:\s*[,;]\s*P?\d+)*\]/gi, '')
      .replace(/^[ \t]{0,3}#{1,6}[ \t]+/gm, '')
      .replace(/^[ \t]*[-*+][ \t]+/gm, '• ')
      .replace(/\*\*([^\n]+?)\*\*|__([^\n]+?)__/g, (_, stars, underscores) => stars || underscores)
      .replace(/(^|[\s(])\*([^\s*](?:[^*\n]*[^\s*])?)\*(?=$|[\s.,!?:;)])/g, '$1$2')
      .replace(/`([^`\n]+)`/g, '$1')
      .replace(/[ \t]+$/gm, '')
      .trim();
  }

  function render(ai, document) {
    const panel = document.getElementById('aiPanel');
    panel.hidden = !ai || ai.status === 'disabled';
    if (panel.hidden) return;
    const pending = ai.status === 'pending';
    panel.setAttribute('aria-busy', String(pending));
    document.getElementById('aiBasis').textContent = pending ? 'Preparing an explanation…'
      : ai.basis === 'general_knowledge' && ai.document_loaded === false ? 'General AI knowledge · No dataset loaded'
      : labels[ai.basis] || 'AI explanation unavailable';
    document.getElementById('aiText').textContent = pending ? 'Generating an AI explanation…'
      : ai.status === 'success' ? explanationText(ai.answer) : ai.message || 'AI is temporarily unavailable. The retrieved result is still available.';
    document.getElementById('aiDetails').textContent = ai.status === 'success' && Number.isFinite(ai.latency_ms)
      ? `${(ai.latency_ms / 1000).toFixed(1)} s`
      : '';
  }

  function sources(result, document) {
    const container = document.getElementById('retrievalSources');
    container.replaceChildren();
    const sources = result.sources || (result.passage_text ? [result] : []);
    for (const source of sources) {
      const text = source.passage_text || source.source_passage;
      if (!text) continue;
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      const passage = document.createElement('p');
      const pid = source.passage_id || source.passage_index;
      summary.textContent = `Source passage ${pid == null ? '' : pid}`;
      passage.textContent = text;
      details.append(summary, passage);
      container.append(details);
    }
  }

  const api = { askPayload, bindToggle, ask, render, sources, explanationText };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AIView = api;
})(typeof window !== 'undefined' ? window : globalThis);
