/*
 * Shim so jspdf-autotable's CommonJS branch (require("jspdf")) resolves to
 * the vendored UMD build when the report tests run in Node.
 */
module.exports = require("../../../static/vendor/jspdf.umd.min.js");
