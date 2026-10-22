# File-level change and validation record

Paths below are relative to the application directory `qa_system/`. Original copies are retained in this folder's `original/`. Verification outputs are in `evidence/`; `IMPLEMENTATION_REPORT.md` explains metrics and limitations.

| File | Why it changed | Verification |
|---|---|---|
| `.gitignore` | Exclude prepared model artifacts and local instance data. | Integrity inventory; no dependency or frontend changes. |
| `app.py` | Validate schemas and limits; secret/debug defaults; session admission/locking; history, health, provenance, full ingestion timing and cached coverage. Distinguish unsupported formats (415) from extraction failures (422). | Backend quality and bug-fix tests; 36 live checks, including eight simultaneous asks and all seven formats. |
| `ingestion.py` | Preserve JSON ownership/scalars, DOCX body order, UTF BOMs and literal CSV values; reject malformed CSV/oversized streams/expanded archives. | Ingestion quality tests; actual HTTP upload and question for TXT, MD, PDF, DOCX, PPTX, CSV, JSON; malformed/UTF-16 probes. |
| `preprocessing.py` | Deterministic tokens retain polarity/numbers/identifiers; bound undelimited passages and corpus work; explicit NLTK preparation. | Minimal-pair/encoding/window/limit tests; live 141,689-character document. |
| `retrievers.py` | Preserve single tokens in sparse vocabulary; share/lock local embedding and reranker models; recover failed loads. | Retriever integration/interface tests and both-mode fixed evaluation suites. |
| `qa_model.py` | Use local SQuAD2 reader and null comparison; multiple source candidates; bounded inference, retryable failures, honest score types; source-aligned summary proposals. | Real-model meaning and offset tests; full fixed evaluations; summary integration; unavailable-model evaluation. |
| `qa_engine.py` | Delegate evidence answering; constrain meta intents; source-only explanations, bounded centrality, accurate timing and expandable suggestions; only expand TF-IDF acronym when present in source passage. | Definitions, offsets, suggestions, backend quality tests and fixed evaluations. |
| `answering.py` (new) | Semantic candidate selection, clause ranking, property/type/polarity/qualifier checks, dated facts, supported composition, conflict and clarification paths. | Meaning tests and 98 fixed cases in each retrieval mode; independent first-pass histories retained. |
| `verifier.py` (new) | Supported grammatical claim conversion, contiguous-source proof and local NLI diagnostics; no truth-probability claim. | Real model meaning/absent-property regressions; development verifier ablation and fixed evaluations. |
| `models.py` (new) | Centralize configured model identities and prepared local paths. | Model loading integrations and 29 prepared-file SHA256 checks. |
| `prepare_models.py` (new) | Explicit download step, four pinned default revisions, preparation/individual-file manifests. | Actual model preparation logs and hashes; pins match prepared revisions; AST/import checks. Optional summarizer uses recorded upstream revision rather than a hardcoded default pin. |
| `summary_engine.py` | Enforce character/token input budgets; reject ungrounded paraphrases; preserve complete coverage in synchronized cache. | Executive summary unit/integration tests; real HTTP four-bullet source-containment and cache checks. |
| `simplifier.py` | Remove context-free individual/individuals substitutions that changed meaning. | Backend simplification regression. |
| `README.md` | Current offline preparation/run instructions, API contracts, score meanings, actual interface limits, model attribution and deployment limitations. | Checked against code, actual model manifests and verified API behavior. |
| `tests/test_bug_fixes.py` | Invalid numeric/empty question contracts now require 400 before counters change. | Full 113-test suite. |
| `tests/test_retrievers.py` | Preserve expected status/source and unchanged original fixture; independently check selected-passage cosine and exact offsets instead of stale route-dependent numbers. | Full 113-test suite and source fixture hash check. |
| `tests/test_suggestions.py` | Source-copy answer confidence is null, with explicit source evidence, rather than fabricated 100%. | Full 113-test suite. |
| `tests/test_backend_quality.py` (new) | API schemas, finite thresholds, capacity, coverage caching, intents, suggestions, deterministic routes, summary and simplification regressions. | Full 113-test suite. |
| `tests/test_ingestion_quality.py` (new) | Representation, parsing and work-limit regressions from verified audit failures. | Full 113-test suite. |
| `tests/test_answer_meaning.py` (new) | Actual neural tests for paraphrases, absent properties, conflicts, all procedure steps, multipart provenance, negative propositions and same-paragraph distractors. | Full 113-test suite with prepared models; no mocked neural-accuracy claims. |

No frontend files, frontend tests, original data fixtures or pinned requirement files were changed. The frozen UI's history/export integration and shared persistent state remain deferred, as described in the implementation report.

The outer Phase 2 folder also contains preservation/freezing scripts, development probes, frozen evaluation sets, the evaluation runner, live server harness, integrity verifier and report generator. These are audit/reproduction artifacts, not runtime application dependencies. Preserve first-pass evidence when running another evaluation; use a new output label.
