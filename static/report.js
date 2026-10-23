/*
 * report.js — Export Q&A Report: pure functions over the session log.
 *
 *   computeStats(records)          → dashboard-matching summary numbers
 *   buildRows(records, timeFmt)    → CSV table rows
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
    "→": "->", "←": "<-", "·": "-", "•": "-", "…": "...", "\u00a0": " ",
  };

  function pdfSafe(text) {
    var s = text == null ? "" : String(text);
    s = s.replace(/[—–’‘“”→←·•…\u00a0]/g, function (ch) { return CHAR_MAP[ch] || " "; });
    // eslint-disable-next-line no-control-regex
    return s.replace(/[^\x09\x0a\x0d\x20-\xff]/g, "");
  }

  function buildPdf(records, meta, jsPDFCtor) {
    var doc = new jsPDFCtor({ orientation: "portrait", unit: "mm", format: "a4" });
    var W = doc.internal.pageSize.getWidth(), H = doc.internal.pageSize.getHeight();
    var margin = 20, bottom = H - 22, y = margin;
    var formatExplanation = meta && meta.formatExplanation || function (text) { return String(text); };
    doc.setTextColor(0);

    function nextPage() {
      doc.addPage();
      y = margin;
    }
    function write(text, size, bold) {
      doc.setFont("helvetica", bold ? "bold" : "normal");
      doc.setFontSize(size);
      var lines = doc.splitTextToSize(pdfSafe(text), W - margin * 2);
      var leading = size * 0.352778 * 1.45;
      lines.forEach(function (line) {
        if (y + leading > bottom) nextPage();
        doc.text(line, margin, y);
        y += leading;
      });
    }
    function section(label, text) {
      // Keep each heading with the opening lines of its content.
      if (y + 23 > bottom) nextPage();
      write(label, 11, true);
      y += 1;
      write(text, 11, false);
      y += 5;
    }

    write("QNext - Q&A Report", 18, true);
    y += 3;
    var created = meta && meta.exportDate ? meta.exportDate : new Date().toLocaleString();
    write("Exported: " + created, 10, false);
    write("Questions: " + records.length, 10, false);
    y += 9;
    if (!records.length) write("No questions in this session.", 11, false);
    records.forEach(function (record, index) {
      var result = record.result || record;
      var retrieval = result.retrieval || result;
      var ai = result.ai;
      var explanation = "AI explanations were off for this question.";
      if (ai && ai.status === "success" && ai.answer) explanation = formatExplanation(ai.answer);
      else if (ai && ai.status === "pending") explanation = "AI explanation is still being generated.";
      else if (ai && ai.status !== "disabled") explanation = "AI explanation was unavailable for this question.";
      var retrievedText = retrieval.answer || (retrieval.reason === "no_dataset"
        ? "No dataset was loaded. This question used general AI knowledge."
        : "No retrieved text was available for this question.");
      doc.setFont("helvetica", "normal");
      doc.setFontSize(11);
      var leading = 11 * 0.352778 * 1.45;
      var height = 3 * (leading + 6) + 5;
      [record.question || "", retrievedText, explanation].forEach(function (text) {
        height += doc.splitTextToSize(pdfSafe(text), W - margin * 2).length * leading;
      });
      // Keep short Q&A entries together; longer entries continue across pages.
      if (y + 38 > bottom || (height <= bottom - margin && y + height > bottom)) nextPage();
      section("Question " + (index + 1), record.question || "");
      section("Retrieved Text", retrievedText);
      section("AI Explanation", explanation);
      y += 5;
    });
    var total = doc.internal.getNumberOfPages();
    for (var page = 1; page <= total; page++) {
      doc.setPage(page);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(9);
      doc.text("Page " + page + " of " + total, W / 2, H - 12, { align: "center" });
    }
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
