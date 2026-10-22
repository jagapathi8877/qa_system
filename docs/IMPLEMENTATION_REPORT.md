# Phase 2 implementation and verification — 22 October 2026

The backend now checks question meaning through relevance ranking, a reader trained for unanswerable questions, source-clause constraints and textual entailment for supported question forms. A shared word or a literal quote alone does not certify an answer. The implementation addresses the reported symptom: returning a related sentence with the wrong property, polarity, date, unit or entity.

## Measured results

| Test set and stage | TF-IDF mode | Semantic mode |
|---|---:|---:|
| Original audit baseline, 32 questions | 22/32 | 21/32 |
| Final audit regressions, 32 questions | 32/32 | 32/32 |
| Original code on new domains, 42 questions | 27/42 | 27/42 |
| First candidate on new domains, 42 questions | 36/42 | 36/42 |
| Final new-domain regressions, 42 questions | 42/42 | 42/42 |
| Original code on fresh final set, 24 questions | 19/24 | 19/24 |
| First candidate on fresh final set, 24 questions | 23/24 | 23/24 |
| Final repaired regressions, 24 questions | 24/24 | 24/24 |

These numbers are **complete required-fact / safe-challenge screens**, not general semantic accuracy, EM, F1 or production-readiness metrics. Positive cases require an answered result containing every predefined required fact; challenge cases require refusal/clarification, or a predefined explicit negative correction. The evaluator also checks exact answer offsets. All final source-offset checks have zero errors. Raw source passages, spans, methods, model diagnostics and timing are preserved in `evidence/validated_*_responses.json`.

The 32-question audit corpus was a development set. The 42-question set was frozen before implementation and its first candidate scored 36/42; its misses then informed repairs, so later scores are regression results. The separate 24-question final set was frozen before its first run and scored 23/24 first pass. F09 was unnecessarily refused as conflicting; the explicit negative proposition path and negation qualifier rule were repaired. Subsequent 24/24 is a regression result, not untouched holdout performance. Neither set was edited after freezing; SHA256 files and integrity checks verify this. The same author created expected facts and reviewed selected raw answers; there was no independent blinded human rating or external benchmark. The substring screen can miss subtle semantic errors; it must not be presented as universal correctness.

With QA, embeddings and reranking explicitly forced unavailable, the fallback answered 17/24 positives completely and handled 8/8 challenges safely. This run is separately labeled, not presented as neural inference. It deliberately abstains on relations and paraphrases it cannot verify. The original audited fallback completed 16/24 positives and safely handled 5/8 challenges.

## Verification and operational checks

- Full Python suite: **113 passed**, including real reader, embedding and summarizer integration; no test failures/errors. Three upstream tokenizer future warnings remain (`validated_pytest.xml` and log).
- Original frontend Node suite: **21 passed**, zero failures (`final_node.log`).
- Real loopback HTTP: **36 checks passed**, including UTF-16/nested-JSON uploads, malformed-CSV rejection, source-grounded real summary, cached coverage, invalid-input counters, independent sessions, 55 empty readers, eight concurrent same-session asks, history attribution and reset. The owned temporary server was terminated by the harness.
- Long input: **141,689 characters** without sentence delimiters ingested in **0.579 seconds** and produced **181 bounded passages**. A missing telephone-number question abstained. This is one synthetic workload, not a load-testing guarantee.
- Real executive summary returned 4 bullets; every bullet was checked for source containment. First and cached calls retained identical coverage metadata.
- Integrity: 37 original snapshot files, 30 Phase 1 evidence/report files and 29 prepared-model files verified. Frontend, original data fixtures and dependency pins remained byte-identical.

Latency observations for final audit questions:

- Audit tfidf: mean 652 ms; median 539 ms; p95 1430 ms.
- Audit semantic: mean 644 ms; median 600 ms; p95 1353 ms.

These are observed local CPU timings, with model cold load outside most asks and some other verification jobs running concurrently. They are not controlled speedup comparisons. Meaning verification adds inference work; quality was prioritized over minimizing latency. Peak memory and production throughput were not rebenchmarked. Phase 1 resource numbers must not be relabeled as measurements of this backend.

## What changed

- `answering.py`: bounded candidate pooling, query–clause relevance, explicit scope/type/polarity checks, null-aware evidence selection, conflict/clarification paths and source-based multi-evidence composition. Conflicts may involve distinct sentences in one passage.
- `verifier.py`: grammatical question-to-claim conversion for supported forms, contiguous source proof and optional NLI. Unknown forms retain explicit diagnostics; the system does not claim universal entailment verification. NLI itself was observed accepting wrong units/subjects, so independent constraints remain necessary.
- `qa_model.py`, `models.py`, `prepare_models.py`, `retrievers.py`: offline model loading, pinned default reader/embedding/reranker/verifier revisions, preparation manifests/hashes, bounded CPU execution, shared locks and recoverable failures. Existing Python dependency pins were retained.
- `ingestion.py`, `preprocessing.py`: ordered document parsing, strict CSV, hierarchical JSON and BOM-aware text; deterministic meaning-preserving tokens; bounded passages and input limits.
- `qa_engine.py`, `summary_engine.py`, `simplifier.py`: source-aligned summaries, scalable centrality, truthful coverage/cache handling, scoped intents and removal of unsafe word substitutions.
- `app.py`: validated JSON API, random/configured signing secret, debug off, admission without silent active eviction, session synchronization, complete ingestion timing and backend history/health.
- Backend tests: new parsing/API/meaning regressions. Three existing backend test files changed because public contracts now reject invalid questions, avoid fabricated source-copy probability and return selected-passage scores. Original fixtures and frontend tests were preserved.

## Audit issue dispositions

| Audit ID | Area | Disposition |
|---|---|---|
| AUD-01 | Signing secret | Random development key; production requires private configured key. |
| AUD-02 | Debugging | Debug off by default; explicit opt-in retained. |
| AUD-03 | API validation | Object schemas, finite thresholds, mode/question limits; invalid calls excluded from counters. |
| AUD-04 | JSON associations | Nested paths and array indices retained. |
| AUD-05 | CSV loss | Strict quoting/shape validation; literal None retained. |
| AUD-06 | UTF-16 | BOM decoding; binary/NUL rejection. |
| AUD-07 | Normalization | Negation, signed/decimal numbers and single-character identifiers retained. |
| AUD-08 | Wrong relation | Semantic reranking, null-aware reading, property/type checks; conservative fallback. |
| AUD-09 | Incomplete answers | Limited multipart/comparison composition, complete procedure evidence, bounded relation chains. Arbitrary reasoning remains unsupported. |
| AUD-10 | Absent/ambiguous answers | Entailment for supported forms and candidate alternatives; measured abstention regressions. |
| AUD-11 | Dates/conflicts/references | Actual answer-clause years, dated-current selection, conflicting spans and unresolved-reference refusals. Universal contradiction detection is not claimed. |
| AUD-12 | Summary fabrication | Only source-aligned sentences accepted; free paraphrases rejected. |
| AUD-13 | Confidence | Reader score explicitly uncalibrated; deterministic routes use null; retrieval score describes selected source. No statistical calibration claim. |
| AUD-14 | Unbounded passage | Hard character windows, overlap and corpus/passage limits. |
| AUD-15 | Coverage metadata | Coverage and warnings retained on cache hits; capped analysis sampling disclosed. |
| AUD-16 | Summary token budget | Individual inputs and reduction bounded by character/token budgets. |
| AUD-17 | Session eviction | Read-only empty sessions allocate no capacity; loaded-session admission rejects overflow, with idle expiry. |
| AUD-18 | Process-local state | Documented limitation; per-session synchronization added. Persistence/shared multi-worker state deferred. |
| AUD-19 | UI/document mismatch | README corrected to actual API/interface. Missing frontend controls remain outside the frozen frontend scope. |
| AUD-20 | Reload/export history | Backend history endpoint added. Frontend restoration/export integration deferred. |
| AUD-21 | Replacement attribution | Per-corpus source identity in backend responses/history. Frozen browser export can still mix older questions under replacement filename. |
| AUD-22 | Simplification routes | Definition/source routes apply requested simplification; no claim of arbitrary semantic rewriting. |
| AUD-23 | Stale numeric snapshots | Original fixture preserved. Assertions now independently verify selected-passage cosine and source/span invariants. |
| AUD-24 | NLTK variability | Fixed normalization and explicit download opt-in. |
| AUD-25 | Timing scope | Full ingestion/index timing and separate retrieval-index timing. |
| AUD-26 | Suggestion capacity | Larger later requests expand cached suggestions. |
| AUD-27 | JSON scalars | Top-level scalar values retained. |
| AUD-28 | Scoped meta intents | Whole-document intents anchored; scoped summary/budget/retention queries remain factual. |
| AUD-29 | Load recovery | Local failed loads retry after bounded cooldown; reader detects preparation changes. |
| AUD-30 | DOCX ordering | Paragraph/table XML body order and merged cells preserved. |
| AUD-31 | Unsafe replacements | Individual/individuals substitutions removed. |

## Remaining practical limits

Frontend source/confidence/retriever controls and history/export integration were frozen and remain deferred. State is ephemeral and single-process, not shared across workers or restarts. Uploading another file replaces the corpus; multi-document storage is not implemented. OCR, arbitrary multi-hop reasoning, conversation memory, statistical confidence calibration and universal contradiction detection are not supported. Summaries intentionally copy grounded source sentences and may omit material under disclosed sampling limits. A source can itself be false. This is a measured local quality improvement, not a guarantee of flawless answers or a production-deployment certification.

## Reproduce

Use Python 3.12 and the pinned requirements. From the application directory run `python prepare_models.py`, optionally `python prepare_models.py --models summarizer`, then `python app.py`. The tested environment in this checkout is the outer `audit/.venv`. Current setup, API and model attribution are in `qa_system/README.md`.

From the outer project directory, `python phase2/2026-10-22-quality/run_matrix.py <new-label>` runs the three fixed suites and separately forced fallback. `live_check.py` starts/stops its own port-5058 server; run only when that port is unused. `verify_integrity.py` verifies immutable evidence and prepared-model hashes. Do not rerun the preparation/freezing scripts over existing baselines. No Git repository was available, and no commit/push/deployment was performed.

Official model documentation: [SQuAD2 reader](https://huggingface.co/deepset/minilm-uncased-squad2), [sentence embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [relevance reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), [textual inference](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768). Model-card claims were not counted as our own measured results.
