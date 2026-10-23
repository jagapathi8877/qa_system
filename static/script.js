/* ==========================================================================
   QNext — frontend logic. All server calls go through fetch() to the
   Flask REST API; nothing on this page ever reloads.
   ========================================================================== */

"use strict";

const $ = (id) => document.getElementById(id);

let chart = null;

const state = {
  datasetLoaded: false,
  busy: false,
  summaryBusy: false,
  exportBusy: false,
  retriever: "tfidf",
  aiMode: true,
  datasetRevision: 0,
  log: [], // SessionRecords for the export report
  sessionQuestionCount: 0,
  datasetSource: null,
  recent: [], // last few {question, answered}
};

/* ------------------------------ tiny helpers ----------------------------- */

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

function fmtMs(ms) {
  if (ms == null) return "—";
  return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms";
}

let toastTimer = null;
function toast(message, ms = 3200) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  el.classList.remove("toast-out");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.add("toast-out");
    setTimeout(() => (el.hidden = true), 240);
  }, ms);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.message || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

/* ------------------------------- dataset pill ---------------------------- */

function setDatasetPill(loaded, source) {
  const pill = $("datasetPill");
  if (loaded) {
    pill.className = "pill pill-loaded";
    const label = source ? `Dataset: ${source}` : "Dataset loaded";
    $("statusText").textContent = label;
    pill.title = label;
  } else {
    pill.className = "pill pill-empty";
    $("statusText").textContent = "No dataset loaded";
    pill.title = "No dataset loaded";
  }
}

function showIngestStatus(kind, html) {
  const el = $("ingestStatus");
  el.className = `ingest-status ${kind}`;
  el.innerHTML = html;
  el.hidden = false;
}

/* --------------------------------- tabs ---------------------------------- */

document.querySelectorAll(".seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document
      .querySelectorAll(".seg-btn")
      .forEach((b) => b.classList.toggle("is-active", b === btn));
    document.querySelectorAll(".tab-body").forEach((body) => {
      body.classList.toggle("is-active", body.dataset.body === btn.dataset.tab);
    });
  });
});

/* ------------------------------- dropzone -------------------------------- */

const dropzone = $("dropzone");
const fileInput = $("fileInput");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-drag");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-drag");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) uploadFile(fileInput.files[0]);
  fileInput.value = "";
});

/* ---------------------------- ingest actions ----------------------------- */

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  showIngestStatus(
    "ok",
    `<strong>Indexing ${escapeHTML(file.name)}…</strong><span class="stat-row">Extracting text and building the TF-IDF index.</span>`
  );
  try {
    const data = await fetch("/api/upload", { method: "POST", body: fd }).then(async (r) => {
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.message || "Upload failed");
      return body;
    });
    onIngested(data);
  } catch (err) {
    showIngestStatus("err", `<strong>Could not index that file.</strong><span class="stat-row">${escapeHTML(err.message)}</span>`);
  }
}

async function pasteText() {
  const text = $("pasteText").value.trim();
  if (!text) return toast("Paste some text first.");
  $("pasteBtn").disabled = true;
  try {
    const data = await api("/api/paste", { method: "POST", body: JSON.stringify({ text }) });
    onIngested(data);
  } catch (err) {
    showIngestStatus("err", `<strong>Could not index that text.</strong><span class="stat-row">${escapeHTML(err.message)}</span>`);
  } finally {
    $("pasteBtn").disabled = false;
  }
}

async function loadSample() {
  $("sampleBtn").disabled = true;
  try {
    const data = await api("/api/sample", { method: "POST" });
    onIngested(data);
    toast("Sample dataset indexed — try a question!");
  } catch (err) {
    toast(err.message);
  } finally {
    $("sampleBtn").disabled = false;
  }
}

function onIngested(data) {
  state.datasetRevision += 1;
  state.datasetLoaded = true;
  state.datasetSource = data.source;
  $("answerCard").hidden = true;
  $("aiPanel").hidden = true;
  $("emptyState").hidden = false;
  $("emptyState").style.display = "";
  setDatasetPill(true, data.source);
  showIngestStatus(
    "ok",
    `<strong>${escapeHTML(data.message)}</strong>
     <span class="stat-row">Source: ${escapeHTML(data.source)} · ${data.chars.toLocaleString()} characters ·
     indexed in ${fmtMs(data.index_time_ms)}</span>`
  );
  setSummaryEnabled(true);
  hideExecSummary();
  loadSuggestions();
}

/* --------------------------- executive summary --------------------------- */

function setSummaryEnabled(loaded) {
  $("summaryBtn").disabled = !loaded || state.summaryBusy;
}

function hideExecSummary() {
  $("execSummary").hidden = true;
}

async function generateSummary() {
  if (!state.datasetLoaded) return toast("Load a dataset first.");
  if (state.summaryBusy) return;
  state.summaryBusy = true;
  setSummaryEnabled(true); // keeps disabled while busy

  const btn = $("summaryBtn");
  btn.classList.add("btn-busy");
  $("summaryBtnLabel").textContent = "Summarizing…";
  hideExecSummary();

  try {
    const data = await api("/api/summarize", { method: "POST" });
    if (data.status === "too_short") {
      toast(data.message, 4500);
      return;
    }
    const list = $("execBullets");
    list.innerHTML = data.bullets
      .map((b) => `<li>${escapeHTML(b)}</li>`)
      .join("");
    const note = $("execNote");
    const bits = [];
    if (data.cached) bits.push("from cache");
    bits.push(`${(data.response_time_ms / 1000).toFixed(1)} s`);
    if (data.coverage_note) bits.push(data.coverage_note);
    note.textContent = bits.join(" · ");
    note.hidden = false;
    $("execSummary").hidden = false;
  } catch (err) {
    toast(err.message, 5000);
  } finally {
    state.summaryBusy = false;
    btn.classList.remove("btn-busy");
    $("summaryBtnLabel").textContent = "Generate executive summary";
    setSummaryEnabled(state.datasetLoaded);
  }
}

$("summaryBtn").addEventListener("click", generateSummary);

$("copySummaryBtn").addEventListener("click", async () => {
  const bullets = [...document.querySelectorAll("#execBullets li")]
    .map((li) => "• " + li.textContent)
    .join("\n");
  if (!bullets) return;
  try {
    await navigator.clipboard.writeText(bullets);
    toast("Executive summary copied to clipboard.");
  } catch (_) {
    toast("Copy failed — select the text manually.");
  }
});

$("pasteBtn").addEventListener("click", pasteText);
$("sampleBtn").addEventListener("click", loadSample);

/* ------------------------------- ask flow -------------------------------- */

const askForm = $("askForm");
const questionInput = $("questionInput");
AIView.bindToggle(document, (enabled) => {
  state.aiMode = enabled;
  if (!state.aiMode) $("aiPanel").hidden = true;
});

/* ------------------------------ theme (dark mode) ------------------------ */

const THEME_KEY = "quillqa-theme";

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem(THEME_KEY, theme); } catch (_) {}
  const btn = $("themeBtn");
  const dark = theme === "dark";
  btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
  refreshChartTheme();
}

$("themeBtn").addEventListener("click", () => {
  applyTheme(currentTheme() === "dark" ? "light" : "dark");
});
applyTheme(currentTheme()); // sync button label/icon with the head-set theme

/* ------------------------------ dashboard drawer ------------------------- */

const dashDrawer = $("dashDrawer");
let dashDrawerOpener = null;

function openDashDrawer() {
  dashDrawer.hidden = false;
  document.body.classList.add("modal-open");
  dashDrawerOpener = document.activeElement;
  $("dashClose").focus();
  ensureChart();
  refreshStats().catch(() => {});
}

function closeDashDrawer() {
  if (dashDrawer.hidden) return;
  dashDrawer.hidden = true;
  document.body.classList.remove("modal-open");
  if (dashDrawerOpener && document.contains(dashDrawerOpener)) dashDrawerOpener.focus();
  dashDrawerOpener = null;
}

$("dashBtn").addEventListener("click", openDashDrawer);
$("dashClose").addEventListener("click", closeDashDrawer);
dashDrawer.addEventListener("click", (e) => {
  if (e.target === dashDrawer) closeDashDrawer();
});

askForm.addEventListener("submit", (e) => {
  e.preventDefault();
  askQuestion();
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.isComposing) {
    e.preventDefault();
    askForm.requestSubmit();
  }
});

/* ------------------------------ voice input ------------------------------ */

const AUTO_SUBMIT_AFTER_VOICE = true;

const askWrap = $("askWrap");
const micBtn = $("micBtn");
let voiceMeterStream = null;
let voiceMeterContext = null;
let voiceMeterFrame = null;
let voiceMeterRequest = 0;

const speechCtl = QuillSpeech.createSpeechInput({
  lang: QuillSpeech.LANG,
  autoSubmit: AUTO_SUBMIT_AFTER_VOICE,
  onStateChange: updateMicState,
  onInterim: (text) => {
    questionInput.value = text; // interim results stream into the input
  },
  onFinal: (text) => {
    questionInput.value = text; // final result replaces the interim text
    if (AUTO_SUBMIT_AFTER_VOICE) askForm.requestSubmit();
  },
  onError: (err) => {
    if (err.message) toast(err.message); // short, non-blocking
  },
});

async function startVoiceMeter() {
  const requestId = ++voiceMeterRequest;
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !AudioContextCtor) {
    toast("Live microphone animation is not supported in this browser.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (requestId !== voiceMeterRequest || !speechCtl.listening) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }

    voiceMeterStream = stream;
    const context = new AudioContextCtor();
    voiceMeterContext = context;
    await context.resume();
    if (requestId !== voiceMeterRequest || !speechCtl.listening) {
      stream.getTracks().forEach((track) => track.stop());
      voiceMeterStream = null;
      voiceMeterContext = null;
      await context.close();
      return;
    }

    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    context.createMediaStreamSource(stream).connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);
    let smoothedLevel = 0;

    const measure = () => {
      if (requestId !== voiceMeterRequest) return;
      analyser.getByteTimeDomainData(samples);
      let sumSquares = 0;
      for (const sample of samples) {
        const amplitude = (sample - 128) / 128;
        sumSquares += amplitude * amplitude;
      }
      const rms = Math.sqrt(sumSquares / samples.length);
      const targetLevel = Math.min(1, Math.max(0, (rms - 0.008) * 12));
      smoothedLevel += (targetLevel - smoothedLevel) * 0.3;
      askWrap.style.setProperty("--voice-glow-opacity", (0.2 + smoothedLevel * 0.75).toFixed(3));
      askWrap.style.setProperty("--voice-glow-scale", (0.75 + smoothedLevel * 0.5).toFixed(3));
      askWrap.style.setProperty("--voice-glow-height", (8 + smoothedLevel * 12).toFixed(1) + "px");
      voiceMeterFrame = requestAnimationFrame(measure);
    };
    measure();
  } catch (err) {
    if (requestId === voiceMeterRequest && speechCtl.listening) {
      stopVoiceMeter();
      toast("Could not access microphone levels — check microphone permission.");
    }
  }
}

function stopVoiceMeter() {
  voiceMeterRequest++;
  if (voiceMeterFrame !== null) cancelAnimationFrame(voiceMeterFrame);
  voiceMeterFrame = null;
  if (voiceMeterStream) {
    voiceMeterStream.getTracks().forEach((track) => track.stop());
    voiceMeterStream = null;
  }
  if (voiceMeterContext) {
    const context = voiceMeterContext;
    voiceMeterContext = null;
    context.close();
  }
  askWrap.style.removeProperty("--voice-glow-opacity");
  askWrap.style.removeProperty("--voice-glow-scale");
  askWrap.style.removeProperty("--voice-glow-height");
}

function updateMicState(stateName) {
  const listening = stateName === "listening";
  askWrap.classList.toggle("is-listening", listening);
  askForm.classList.toggle("is-listening", listening);
  if (!listening) stopVoiceMeter();
  micBtn.setAttribute("aria-pressed", String(listening));
  micBtn.setAttribute(
    "aria-label",
    listening ? "Stop listening" : "Ask by voice"
  );
  micBtn.title = listening ? "Stop listening" : "Ask by voice";
}

if (speechCtl.supported) {
  micBtn.hidden = false;
  micBtn.addEventListener("click", () => {
    if (speechCtl.listening) {
      speechCtl.stop();
    } else if (speechCtl.start()) {
      startVoiceMeter();
    }
  });
} else {
  // Unsupported (e.g. Firefox): no dead button — disabled with a tooltip.
  micBtn.hidden = false;
  micBtn.disabled = true;
  micBtn.title = "Voice input is not supported in this browser — try Chrome or Edge";
  micBtn.setAttribute("aria-disabled", "true");
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (speechCtl.listening) speechCtl.stop();
    if (!dashDrawer.hidden) closeDashDrawer();
  }
});
window.addEventListener("pagehide", () => speechCtl.abort());

function setRetriever(mode) {
  state.retriever = mode === "semantic" ? "semantic" : "tfidf";
}


/* Chip click handling */
const chipsWrap = $("exampleChips");
if (chipsWrap) {
  chipsWrap.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) {
      questionInput.value = chip.textContent.trim();
      askQuestion();
    }
  });
}

async function loadSuggestions() {
  if (!state.datasetLoaded) return;
  try {
    const data = await api("/api/suggest");
    if (data.status === "ok" && Array.isArray(data.questions) && data.questions.length > 0) {
      const wrap = $("exampleChips");
      if (wrap) {
        wrap.innerHTML = data.questions
          .map((q) => `<button class="chip" type="button">${escapeHTML(q)}</button>`)
          .join("");
      }
    }
  } catch (_) {
    /* suggestions optional */
  }
}

function setBusy(busy) {
  $("aiMode").disabled = busy;
  state.busy = busy;
  updateExportButtons();
  micBtn.disabled = busy || !speechCtl.supported;
  micBtn.disabled = busy || !speechCtl.supported;
  $("skeleton").hidden = !busy;
  if (busy) {
    $("emptyState").hidden = true;
    $("emptyState").style.display = "none";
    $("answerCard").hidden = true;
    AIView.render(state.aiMode ? { status: "pending" } : { status: "disabled" }, document);
  }
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return toast("Type a question first.");
  if (!state.datasetLoaded && !state.aiMode) return toast("Turn on AI explanations or load a dataset first.");
  if (state.busy) return;
  speechCtl.stop(); // never keep the mic open across a submit

  setBusy(true);
  const revision = state.datasetRevision;
  try {
    const result = await AIView.ask({
      question, retriever: state.retriever, enabled: state.aiMode, api,
      onRetrieval: (result) => { if (revision === state.datasetRevision) renderAnswer(result); },
      onAI: (ai) => { if (revision === state.datasetRevision) AIView.render(ai, document); },
    });
    if (revision !== state.datasetRevision) return;
    logQuestion(question, result);
    state.recent.unshift({
      question,
      answered: result.status === "answered" || result.status === "summarized" || result.ai?.status === "success",
    });
    state.recent = state.recent.slice(0, 4);
    renderRecent();
    await refreshStats();
  } catch (err) {
    $("aiPanel").hidden = true;
    if (err.status === 409) {
      toast(err.message);
      state.datasetLoaded = false;
      setDatasetPill(false);
    } else {
      toast(err.message);
    }
    if ($("answerCard").hidden) {
      $("emptyState").hidden = false;
      $("emptyState").style.display = "";
    }
  } finally {
    setBusy(false);
  }
}

function renderAnswer(result) {
  const card = $("answerCard");
  const answered = result.status === "answered";
  const isSummary = result.status === "summarized";
  const isExplanation = isSummary && result.category === "Explanation";

  $("emptyState").hidden = true;
  $("emptyState").style.display = "none";
  card.hidden = result.reason === "no_dataset";
  card.classList.toggle("is-no", !answered && !isSummary);
  card.classList.toggle("is-summary", isSummary);
  $("skeleton").hidden = true;
  $("answerLabel").textContent = "Retrieved Text";
  AIView.render(result.ai, document);
  AIView.sources(result, document);
  const note = $("retrievalNote");
  note.hidden = result.status !== "no_answer" || !["no_relevant_evidence", "no_explicit_definition"].includes(result.reason);
  note.textContent = "This information was not found by the document retrieval engine. The original retrieval result is shown above.";

  // Highlight the model's extracted span inside the composed answer sentence (case-insensitively).
  const answerEl = $("answerText");
  const span = result.span;
  const ans = result.answer || "";
  const at = answered && span ? ans.toLowerCase().indexOf(span.toLowerCase()) : -1;
  if (at !== -1) {
    const matchLen = span.length;
    answerEl.innerHTML =
      escapeHTML(ans.slice(0, at)) +
      '<mark class="span-hl">' + escapeHTML(ans.slice(at, at + matchLen)) + "</mark>" +
      escapeHTML(ans.slice(at + matchLen));
  } else {
    answerEl.textContent = ans;
  }

  // Simplified key points (document explanation) or a simplified answer
  // ("explain X in simple words").
  const block = $("simpleBlock");
  const points = result.simplified_points;
  if (isSummary && points && points.length) {
    $("simpleTitle").textContent = "Key points — simplified";
    $("simplePoints").innerHTML = points.map((p) => `<li>${escapeHTML(p)}</li>`).join("");
    block.hidden = false;
  } else if (answered && result.answer_simple) {
    $("simpleTitle").textContent = "In simple terms";
    $("simplePoints").innerHTML = `<li>${escapeHTML(result.answer_simple)}</li>`;
    block.hidden = false;
  } else {
    block.hidden = true;
  }

  const cat = $("answerCategory");
  cat.textContent = result.category || "Factual";
  cat.style.display = answered || isSummary ? "" : "none";
}

function renderRecent() {
  const wrap = $("recentList");
  if (!state.recent.length) return;
  wrap.hidden = false;
  $("recentItems").innerHTML = state.recent
    .map(
      (r) =>
        `<li><span class="q">${escapeHTML(r.question)}</span>
         <span class="${r.answered ? "ok" : "no"}">${r.answered ? "✓ answered" : "✕ no info"}</span></li>`
    )
    .join("");
}

/* ------------------------------- dashboard ------------------------------- */

const CATEGORY_COLORS = {
  Definition: "#a86cf0",
  Factual: "#45c8df",
  Procedural: "#f0b84f",
  Comparison: "#ed68b7",
  Summary: "#7590ff",
  Explanation: "#6bcf9e",
};

function chartTheme() {
  const dark = currentTheme() === "dark";
  return { label: dark ? "#c1b5d2" : "#61566f", border: dark ? "#0e0d14" : "#fffbff" };
}

function refreshChartTheme() {
  if (!chart) return;
  const t = chartTheme();
  chart.options.plugins.legend.labels.color = t.label;
  chart.data.datasets[0].borderColor = t.border;
  chart.update();
}

function ensureChart() {
  if (chart) {
    chart.resize();
    return;
  }
  const t = chartTheme();
  chart = new Chart($("categoryChart").getContext("2d"), {
    type: "doughnut",
    data: {
      labels: ["Definition", "Factual", "Procedural", "Comparison", "Summary", "Explanation"],
      datasets: [
        {
          data: [0, 0, 0, 0, 0, 0],
          backgroundColor: Object.values(CATEGORY_COLORS),
          borderColor: t.border,
          borderWidth: 2,
          hoverOffset: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",
      plugins: {
        legend: {
          position: "right",
          labels: { boxWidth: 10, boxHeight: 10, borderRadius: 3, useBorderRadius: true, font: { family: "Inter", size: 11 }, color: t.label },
        },
      },
    },
  });
}

async function refreshStats() {
  const stats = await api("/api/stats");
  $("statTotal").textContent = stats.total_queries;
  $("statAnswered").textContent = stats.answered;
  $("statNoAnswer").textContent = stats.no_answer;
  $("statTime").innerHTML = `${Math.round(stats.avg_response_ms)}<span class="stat-unit">ms</span>`;
  const count = $("dashCount");
  count.textContent = stats.total_queries;
  count.hidden = stats.total_queries === 0;
  if (chart) {
    chart.data.datasets[0].data = [
      stats.categories.Definition || 0,
      stats.categories.Factual || 0,
      stats.categories.Procedural || 0,
      stats.categories.Comparison || 0,
      stats.categories.Summary || 0,
      stats.categories.Explanation || 0,
    ];
    chart.update();
  }
}

/* ------------------------ session log + export --------------------------- */

const exportCsvBtn = $("exportCsvBtn");
const exportPdfBtn = $("exportPdfBtn");

function logQuestion(question, result) {
  // One SessionRecord per ask — the raw material for the export report.
  state.log.push({
    timestamp: new Date().toISOString(),
    question: question,
    category: result.category,
    answer: result.answer,
    confidence: result.confidence,
    latency_ms: result.response_time_ms,
    retrieval_score: result.retrieval_score,
    retriever: result.retriever,
    source_passage_id: result.passage_index,
    status: result.status,
    ai: result.ai,
    corpus_id: result.corpus_id,
    source: result.source,
  });
  updateExportButtons();
}

function updateExportButtons() {
  const hasRows = state.log.length > 0;
  [exportCsvBtn, exportPdfBtn].forEach((btn) => {
    const available = btn === exportPdfBtn ? hasRows || state.sessionQuestionCount > 0 : hasRows;
    btn.disabled = !available || state.exportBusy || state.busy;
    btn.title = available
      ? "Download this session's Q&A report"
      : "Ask a question first — there is nothing to export yet";
  });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

exportCsvBtn.addEventListener("click", () => {
  if (!state.log.length) return;
  const blob = new Blob([Report.buildCsv(state.log)], {
    type: "text/csv;charset=utf-8",
  });
  downloadBlob(blob, Report.stamp() + ".csv");
  toast("CSV report downloaded.");
});

exportPdfBtn.addEventListener("click", async () => {
  if (state.exportBusy || state.busy || typeof Report.buildPdf !== "function") return;
  state.exportBusy = true;
  updateExportButtons();
  const revision = state.datasetRevision;
  try {
    const history = await api("/api/history");
    if (revision !== state.datasetRevision) return;
    const doc = Report.buildPdf(history.records, {
      exportDate: new Date().toLocaleString(),
      formatExplanation: AIView.explanationText,
    }, window.jspdf.jsPDF);
    doc.save(Report.stamp() + ".pdf");
    toast("PDF report downloaded.");
  } catch (err) {
    toast("PDF export failed: " + err.message);
  } finally {
    state.exportBusy = false;
    updateExportButtons();
  }
});

/* ------------------------------ reset session ---------------------------- */

$("resetBtn").addEventListener("click", async () => {
  try {
    await api("/api/dataset", { method: "DELETE" });
    state.datasetRevision += 1;
    state.datasetLoaded = false;
    state.recent = [];
    setDatasetPill(false);
    $("ingestStatus").hidden = true;
    $("pasteText").value = "";
    $("answerCard").hidden = true;
    $("aiPanel").hidden = true;
    $("recentList").hidden = true;
    $("recentItems").innerHTML = "";
    $("emptyState").hidden = false;
    $("emptyState").style.display = "";
    $("answerCard").classList.remove("is-summary");
    $("answerLabel").textContent = "Retrieved Text";
    $("simpleBlock").hidden = true;
    state.log = []; // the report log dies with the session
    state.sessionQuestionCount = 0;
    updateExportButtons();
    setSummaryEnabled(false);
    hideExecSummary();
    await refreshStats();
    $("exampleChips").replaceChildren();
    state.datasetSource = null;
    toast("Session reset — ask with AI or load a new dataset.");
  } catch (err) {
    toast(err.message);
  }
});

/* --------------------------------- boot ---------------------------------- */

(async function boot() {
  try {
    const data = await api("/api/dataset");
    state.datasetLoaded = !!data.loaded;
    setDatasetPill(!!data.loaded, data.source);
  } catch (_) {
    /* first visit — fine */
  }
  setSummaryEnabled(state.datasetLoaded);
  try {
    const history = await api("/api/history");
    state.sessionQuestionCount = history.records.length;
  } catch (_) { /* History can be retried when exporting. */ }
  hideExecSummary();
  updateExportButtons();
  if (state.datasetLoaded) {
    loadSuggestions();
  }
  await refreshStats().catch(() => {});
  questionInput.focus();
})();
