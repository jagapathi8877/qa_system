/*
 * Unit tests for static/report.js — run with:  node --test tests/js/
 *
 * CSV: escaping, injection guard, BOM, header/row counts, round-trip via a
 * minimal parser. Stats: dashboard-matching aggregation. PDF: built with the
 * vendored jsPDF + autotable (a module resolver shim lets autotable's CJS
 * branch find "jspdf"), asserting complete text, monochrome output and pagination.
 */

import test from "node:test";
import assert from "node:assert/strict";
import Module from "node:module";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";
import AIView from "../../static/ai-panel.js";

/* --------------------------------------------------- jspdf resolve shim */
const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...args) {
  if (request === "jspdf") {
    return path.join(here, "helpers", "jspdf-shim.cjs");
  }
  return origResolve.call(this, request, ...args);
};

global.window = global;
try {
  global.navigator = { userAgent: "node" };
} catch (_) {
  // Node ≥21 exposes a read-only global navigator
  Object.defineProperty(global, "navigator", {
    value: { userAgent: "node" },
    configurable: true,
  });
}
const { jsPDF } = require(path.join(here, "..", "..", "static", "vendor", "jspdf.umd.min.js"));
require(path.join(here, "..", "..", "static", "vendor", "jspdf.plugin.autotable.min.js"));

import Report from "../../static/report.js";

/* ------------------------------------------------------------- helpers */

function record(overrides, i = 1) {
  return Object.assign(
    {
      index: i,
      timestamp: "2026-09-30T10:00:00",
      question: "What is machine learning?",
      category: "Definition",
      answer: "Based on the dataset: a branch of artificial intelligence.",
      confidence: 0.5,
      latency_ms: 120,
      retrieval_score: 0.25,
      retriever: "tfidf",
      source_passage_id: 2,
      status: "answered",
    },
    overrides
  );
}

function parseCsvLine(line) {
  // minimal RFC-4180 parser for round-trip assertions
  const cells = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') inQuotes = false;
      else cur += ch;
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      cells.push(cur); cur = "";
    } else {
      cur += ch;
    }
  }
  cells.push(cur);
  return cells;
}

/* ----------------------------------------------------------- CSV tests */

test("buildCsv emits BOM, header, and one row per record", () => {
  const csv = Report.buildCsv([record(), record({ index: 2, status: "no_answer", confidence: 0, retrieval_score: 0 })]);
  assert.ok(csv.startsWith("\uFEFF"));
  const lines = csv.replace(/^\uFEFF/, "").trim().split("\r\n");
  assert.equal(lines.length, 3); // header + 2 rows
  const header = parseCsvLine(lines[0]);
  assert.equal(header[0], "No");
  assert.equal(header[2], "Question");
  assert.equal(header[10], "Status");
});

test("cells with comma, quote, and newline round-trip", () => {
  const tricky = 'She said "run, now"\nthen left';
  const csv = Report.buildCsv([record({ question: tricky })]);
  const dataLine = csv.split("\r\n")[1];
  const cells = parseCsvLine(dataLine);
  assert.equal(cells[2], tricky); // exact round-trip
});

test("formula-leading cells are neutralized", () => {
  for (const prefix of ["=", "+", "-", "@"]) {
    const csv = Report.buildCsv([record({ question: prefix + "calc(1)" })]);
    assert.ok(csv.includes("'" + prefix + "calc(1)"), `missing guard for ${prefix}`);
  }
});

test("normal cells are not quoted or prefixed", () => {
  const csv = Report.buildCsv([record({ question: "plain question" })]);
  assert.ok(csv.includes("plain question"));
  assert.ok(!csv.includes("'plain"));
});

test("summary rows carry null confidence and retrieval score as empty cells", () => {
  const csv = Report.buildCsv([
    record({ status: "summarized", category: "Summary", confidence: null, retrieval_score: null, source_passage_id: null }),
  ]);
  const cells = parseCsvLine(csv.split("\r\n")[1]);
  assert.equal(cells[3], "Summary");
  assert.equal(cells[5], "");
  assert.equal(cells[7], "");
  assert.equal(cells[9], "");
});

/* ---------------------------------------------------------- stats tests */

test("computeStats matches dashboard semantics", () => {
  const records = [
    record({ category: "Definition", confidence: 0.8, latency_ms: 100 }),
    record({ category: "Definition", confidence: 0.6, latency_ms: 200 }),
    record({ status: "no_answer", category: "Factual", confidence: 0, latency_ms: 30 }),
    record({ status: "summarized", category: "Summary", confidence: null, latency_ms: 1000, retrieval_score: null }),
  ];
  const s = Report.computeStats(records);
  assert.equal(s.total, 4);
  assert.equal(s.answered, 3);      // answered + summarized (dashboard tile)
  assert.equal(s.noInfo, 1);
  assert.equal(s.answeredRate, 0.75);
  assert.equal(s.avgConfidence, 0.7); // only extractive answers, (0.8+0.6)/2
  assert.equal(s.avgLatency, 332.5);
  assert.deepEqual(s.categories, { Definition: 2, Factual: 1, Summary: 1 });
});

test("computeStats on empty log", () => {
  const s = Report.computeStats([]);
  assert.equal(s.total, 0);
  assert.equal(s.answeredRate, 0);
  assert.equal(s.avgConfidence, null);
  assert.equal(s.avgLatency, null);
});

/* ----------------------------------------------------------- PDF tests */

test("pdfSafe maps typographic characters", () => {
  assert.equal(Report.pdfSafe("A — B → C · D…"), "A - B -> C - D...");
});

test("pdf with zero rows fits one page", () => {
  const doc = Report.buildPdf([], { datasetName: "Sample", exportDate: "2026-09-30 10:00" }, jsPDF);
  assert.equal(doc.getNumberOfPages(), 1);
});

test("pdf paginates all questions with page numbers for a long session", () => {
  const many = Array.from({ length: 60 }, (_, i) =>
    record({
      question: `Question number ${i} asking about a fairly long topic with several words to force wrapping`,
      answer: `Answer number ${i} with a longer explanation sentence that must wrap across multiple lines inside the table cell without overflowing the page.`,
      index: i + 1,
    })
  );
  const doc = Report.buildPdf(many, { datasetName: "Sample", exportDate: "2026-09-30 10:00" }, jsPDF);
  assert.ok(doc.getNumberOfPages() > 1, "60 rows should span multiple pages");
  // text-extraction check: the title appears on page 1 and the footer marks pages
  const raw = doc.output(); // uncompressed content streams
  assert.ok(raw.includes("QNext - Q&A Report"));
  assert.ok(raw.includes("Question number 0"));
  assert.ok(raw.includes("Question number 59"));
  assert.ok(raw.includes("Page 2 of"));
  assert.ok(doc.internal.pageSize.getHeight() > doc.internal.pageSize.getWidth());
});

test("PDF exports session history as monochrome question, retrieval and AI sections", () => {
  const records = [
    { question: 'What is machine learning?', result: {
      retrieval: { answer: 'Machine learning finds patterns in data.' },
      ai: { status: 'success', model: 'DO-NOT-EXPORT-MODEL',
        answer: '**Learning:** A model learns from examples [P1, P2].' },
    } },
    { question: 'Explain photosynthesis.', result: {
      retrieval: { reason: 'no_dataset', answer: null },
      ai: { status: 'success', answer: 'Plants use light to produce sugars.' },
    } },
    { question: 'Where is the laboratory?', result: {
      retrieval: { answer: 'The laboratory is in London.' }, ai: { status: 'disabled' },
    } },
    { question: 'How does learning work?', result: {
      retrieval: { answer: 'Training uses labelled examples.' },
      ai: { status: 'unavailable', message: 'SECRET-PROVIDER-ERROR' },
    } },
    { question: 'Explain the complete process.', result: {
      retrieval: { answer: 'Original evidence begins here. '.repeat(220) + 'EVIDENCE-END' },
      ai: { status: 'success', answer: 'A detailed explanation with multiple steps. '.repeat(240) + 'EXPLANATION-END' },
    } },
  ];
  const before = structuredClone(records);
  const doc = Report.buildPdf(records, { exportDate: '2026-10-23', formatExplanation: AIView.explanationText }, jsPDF);
  const raw = doc.output();
  for (let i = 1; i <= records.length; i++) assert.ok(raw.includes('Question ' + i));
  for (const text of ['Retrieved Text', 'AI Explanation', 'Learning: A model learns from examples.',
    'No dataset was loaded.', 'AI explanations were off', 'AI explanation was unavailable',
    'EVIDENCE-END', 'EXPLANATION-END']) assert.ok(raw.includes(text), text);
  assert.doesNotMatch(raw, /DO-NOT-EXPORT-MODEL|SECRET-PROVIDER-ERROR|\[P1|\*\*Learning|Avg confidence|Categories:/);
  assert.doesNotMatch(raw, /\d(?:\.\d+)? \d(?:\.\d+)? \d(?:\.\d+)? (?:rg|RG)\b/);
  assert.deepEqual(records, before);
  assert.ok(doc.getNumberOfPages() > 2);
  if (process.env.QNEXT_PDF_PROOF) {
    fs.writeFileSync(process.env.QNEXT_PDF_PROOF, Buffer.from(doc.output('arraybuffer')));
  }
});
