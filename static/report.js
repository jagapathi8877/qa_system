/*
 * report.js — Export Q&A Report: pure functions over the session log.
 *
 *   computeStats(records)          → dashboard-matching summary numbers
 *   buildRows(records, timeFmt)    → table rows shared by CSV and PDF
 *   buildCsv(records)              → CSV string (UTF-8 BOM added by caller's
 *                                    Blob; Excel-safe escaping + injection guard)
 *   buildPdf(records, meta, jsPDF) → jsPDF document (jsPDF ctor injected so
 *                                    the function stays testable)
 *
 * The session log lives in script.js state.log; these functions never touch
 * the DOM, which keeps them unit-testable in Node.
 */

(function (global) {
  "use strict";

  var COLUMNS = [
    "No", "Timestamp", "Question", "Category", "Answer", "Confidence",
    "Latency (ms)", "Retrieval score", "Retriever", "Source passage", "Status",
  ];

  /* --------------------------------------------------------- formatting */

  function fmtTimestamp(iso, style) {
    var d = new Date(iso);
    if (isNaN(d.getTime())) return "";
    var p = function (n) { return String(n).padStart(2, "0"); };
    var date = d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate());
    var time = p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
    return style === "short" ? time : date + " " + time;
  }

  function stamp() {
    var d = new Date();
    var p = function (n) { return String(n).padStart(2, "0"); };
    return "qnext-report-" + d.getFullYear() + p(d.getMonth() + 1) +
           p(d.getDate()) + "-" + p(d.getHours()) + p(d.getMinutes());
  }

  /* ----------------------------------------------------------- CSV bits */

  function csvCell(value) {
    var s = value == null ? "" : String(value);
    // CSV injection guard: neutralize formula-leading characters.
    if (/^[=+\-@]/.test(s)) s = "'" + s;
    // Quote when the cell contains a delimiter, quote, or newline.
    if (/[",\n\r]/.test(s)) s = '"' + s.replace(/"/g, '""') + '"';
    return s;
  }

  function buildRows(records, timeStyle) {
    return records.map(function (r, i) {
      return [
        i + 1,
        fmtTimestamp(r.timestamp, timeStyle),
        r.question || "",
        r.category || "",
        r.answer || "",
        r.confidence == null ? "" : Number(r.confidence).toFixed(4),
        r.latency_ms == null ? "" : r.latency_ms,
        r.retrieval_score == null ? "" : Number(r.retrieval_score).toFixed(4),
        r.retriever || "",
        r.source_passage_id == null ? "" : r.source_passage_id,
        r.status || "",
      ];
    });
  }

  function buildCsv(records) {
    var lines = [COLUMNS.join(",")];
    buildRows(records).forEach(function (row) {
      lines.push(row.map(csvCell).join(","));
    });
    return "\uFEFF" + lines.join("\r\n") + "\r\n"; // BOM so Excel picks UTF-8
  }

  /* -------------------------------------------------------- stats + PDF */

  function mean(list) {
    if (!list.length) return null;
    return list.reduce(function (a, b) { return a + b; }, 0) / list.length;
  }

  function computeStats(records) {
    var total = records.length;
    var answered = records.filter(function (r) { return r.status === "answered"; }).length;
    var summarized = records.filter(function (r) { return r.status === "summarized"; }).length;
    var noInfo = records.filter(function (r) { return r.status === "no_answer"; }).length;
    var categories = {};
    records.forEach(function (r) {
      var c = r.category || "Uncategorized";
      categories[c] = (categories[c] || 0) + 1;
    });
    var confidences = records
      .filter(function (r) { return r.status === "answered" && r.confidence != null; })
      .map(function (r) { return r.confidence; });
    var latencies = records
      .filter(function (r) { return r.latency_ms != null; })
      .map(function (r) { return r.latency_ms; });
    return {
      total: total,
      answered: answered + summarized, // matches the dashboard's "Answered" tile
      noInfo: noInfo,
      answeredRate: total ? (answered + summarized) / total : 0,
      avgConfidence: mean(confidences),
      avgLatency: mean(latencies),
      categories: categories,
    };
  }

  /* jsPDF's built-in fonts are WinAnsi-only — map the common typographic
     characters the UI produces and drop anything outside Latin-1. */
  var CHAR_MAP = {
    "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
    "→": "->", "←": "<-", "·": "-", "…": "...", "\u00a0": " ",
  };

  function pdfSafe(text) {
    var s = text == null ? "" : String(text);
    s = s.replace(/[—–’‘“”→←·…\u00a0]/g, function (ch) { return CHAR_MAP[ch] || " "; });
    // eslint-disable-next-line no-control-regex
    return s.replace(/[^\x09\x0a\x0d\x20-\xff]/g, "");
  }

  function buildPdf(records, meta, jsPDFCtor) {
    var stats = computeStats(records);
    var doc = new jsPDFCtor({ orientation: "landscape", unit: "mm", format: "a4" });
    var W = doc.internal.pageSize.getWidth();

    doc.setFont("helvetica", "bold");
    doc.setFontSize(16);
    doc.text("QNext - Q&A Report", 14, 14);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    var created = meta && meta.exportDate ? meta.exportDate : new Date().toLocaleString();
    var dataset = (meta && meta.datasetName) || "Unknown dataset";
    doc.text("Dataset: " + pdfSafe(dataset), 14, 20);
    doc.text("Exported: " + created, 14, 24.5);

    doc.setFontSize(10);
    var avgConf = stats.avgConfidence == null ? "-" : Math.round(stats.avgConfidence * 100) + "%";
    var avgLat = stats.avgLatency == null ? "-" : Math.round(stats.avgLatency) + " ms";
    var cats = Object.keys(stats.categories)
      .map(function (k) { return k + " " + stats.categories[k]; })
      .join(", ") || "-";
    doc.text(
      "Questions: " + stats.total +
      "   |   Answered: " + stats.answered + " (" + Math.round(stats.answeredRate * 100) + "%)" +
      "   |   No info: " + stats.noInfo +
      "   |   Avg confidence: " + avgConf +
      "   |   Avg latency: " + avgLat,
      14, 30
    );
    doc.text("Categories: " + pdfSafe(cats), 14, 35);

    var rows = buildRows(records, "short").map(function (row) {
      return row.map(pdfSafe);
    });

    doc.autoTable({
      head: [COLUMNS],
      body: rows,
      startY: 39,
      styles: { fontSize: 7.5, cellPadding: 1.4, overflow: "linebreak", font: "helvetica" },
      headStyles: { fillColor: [118, 43, 42], textColor: [255, 249, 239], fontStyle: "bold" },
      alternateRowStyles: { fillColor: [241, 232, 212] },
      columnStyles: {
        0: { cellWidth: 9 },        // No
        1: { cellWidth: 20 },       // Time
        2: { cellWidth: 52 },       // Question (wraps)
        3: { cellWidth: 20 },       // Category
        5: { cellWidth: 16 },       // Confidence
        6: { cellWidth: 15 },       // Latency
        7: { cellWidth: 16 },       // Retrieval score
        8: { cellWidth: 17 },       // Retriever
        9: { cellWidth: 14 },       // Source
        10: { cellWidth: 18 },      // Status
      },
      didDrawPage: function () {
        doc.setFontSize(8);
        doc.setTextColor(120);
        doc.text(
          "QNext session report - page " + doc.internal.getNumberOfPages(),
          14, doc.internal.pageSize.getHeight() - 6
        );
        doc.setTextColor(0);
      },
    });
    return doc;
  }

  var api = {
    COLUMNS: COLUMNS,
    computeStats: computeStats,
    buildRows: buildRows,
    buildCsv: buildCsv,
    buildPdf: buildPdf,
    pdfSafe: pdfSafe,
    csvCell: csvCell,
    fmtTimestamp: fmtTimestamp,
    stamp: stamp,
  };

  global.Report = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
