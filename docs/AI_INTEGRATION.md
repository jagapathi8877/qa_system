# QNext Gemini integration — 23 October 2026

## Architecture and audited contracts

The existing pipeline is unchanged: ordered TXT/MD/PDF/DOCX/PPTX/CSV/JSON ingestion; bounded original-text passages; deterministic NLTK stemming with negation/number preservation; sparse TF-IDF or optional local semantic retrieval; optional cross-encoder ranking; null-aware SQuAD2 reading; explicit relation/type/date/negation constraints and optional textual entailment. `answering.py` preserves exact answer/source offsets. Definitions, suggestions, extractive explanations and executive summaries retain their existing separate routes. Summaries do not increment QA counters. `/api/health` describes the local models; its `network_during_inference: false` describes local NLP only, while explicitly enabled AI calls use Google's API.

There was no existing remote AI abstraction. `app.py` stores each active engine, source, history, statistics and lock in signed-cookie-associated process memory. Replacement uploads swap engines; each ingestion now also receives a new revision, even for identical content. Limits, session admission, local models and original answer statuses remain intact. No database or external response cache was added.

New responsibilities:

| Files | Responsibility |
|---|---|
| `ai_config.py`, `.env.example`, `requirements.txt` | Server-only slots, model capability registry, validated settings, official SDK and dotenv |
| `gemini_gateway.py` | Client reuse, normalization, bounded worker calls, health/cooldown selection and classified failover |
| `ai_explainer.py` | Original evidence selection, prompt boundaries, adaptive output budgets and citation validation |
| `app.py`, `stats.py` | Additive API orchestration, deferred generation, revision checks and separate AI counters |
| `static/ai-panel.js`, `static/script.js`, `templates/index.html`, `static/style.css` | Default-on toggle, progressive independent panels and escaped source/model text |
| `prepare_models.py` | Repair pre-existing optional BART preparation: restricted tensor loading and conversion to safetensors, strict state validation including its tied output embedding |
| `tests/test_ai_*.py`, `tests/test_gemini_gateway.py`, `tests/js/test_ai_panel.mjs`, `tests/data/ai_evaluation.json` | Mocked failure/integration/frontend tests and synthetic evaluation cases |
| `scripts/check_gemini.py`, `scripts/evaluate_ai.py` | Explicit operator checks and repeatable offline/live evaluation |
| `README.md`, this guide, `docs/verification/` | Setup, measured evidence and limitations |

Core ingestion, preprocessing, retrieval, answering, verifier and QA model files, original tests, fixtures, speech, executive-summary code and report generator are preserved. Original source SHA256 values are in `verification/original-files.json`. Previous reports in this directory describe earlier work, not verification of this integration.

## API and interface

```json
{"question":"What is normalization?","retriever":"tfidf","ai_mode":true}
```

`ai_mode` and `defer_ai` must be actual JSON booleans. Omitted `ai_mode` defaults false for existing consumers; the browser explicitly sends its default true. `QNEXT_AI_ENABLED=false` is the server-wide generation kill switch. Old top-level fields retain their values, including retrieval-only `response_time_ms`. `retrieval` is an independent snapshot of those fields; `ai` and `request_id` are additive. Retrieval stats remain separate from AI stats.

The browser sends `defer_ai:true`. `/api/ask` completes and returns the original result plus a pending AI status. It immediately renders Retrieved Text and expandable source passages. `/api/explain` receives only `{request_id, ai_mode:true}` and completes that session's previously authorized question. The backend checks session ownership and active document revision. Duplicate completion returns the saved response without another generation. `ai_mode:false` cancels a pending explanation. A retrieval-only question cannot be upgraded through this endpoint. Missing documents and retrieval errors retain their previous HTTP errors and never invoke Gemini.

Generated explanations render as plain text with preserved line breaks; no model text becomes HTML. The toggle remains disabled during a submission. Document replacement/reset increments the frontend revision and hides stale results; an in-flight earlier response cannot overwrite the new document's view. Provider failure leaves Retrieved Text intact. No-answer messages preserve the engine's exact meaning; conflicting or uncertain evidence is not relabelled as absent.

`ai.status` is `success`, `pending`, `unavailable` or `disabled`. `ai.basis` distinguishes `document_evidence`, `general_knowledge`, `conflicting_evidence`, `incomplete_evidence`, `unavailable`, and `disabled`. Diagnostics contain model ID, slot number, latency, total attempts, failover count, fixed error categories and context truncation. The interface shows useful status text, rather than provider internals. History retains both results. CSV/PDF exports intentionally keep their original retrieved-answer schema; generated explanations are excluded.

## Secure configuration and authorization checks

1. Use Python 3.10–3.12; this verification used 3.12.14 on Windows 11 AMD64. Install `requirements.txt` and `requirements-dev.txt` in a local environment.
2. Copy `.env.example` to `.env`. Empty slots are ignored. Supply `GEMINI_API_KEY_1`, `GEMINI_PROJECT_ID_1` and an opaque `GEMINI_ACCOUNT_LABEL_1` such as `research-a`. Add slots 2 and 3 when authorized. Keys must remain server-side. Never use account emails as diagnostic labels. Environment variables take precedence over `.env`.
3. In Google AI Studio, inspect each actual key's owning project. Match that project with the corresponding Cloud Console project ID. Verify the account's authorization, billing/eligibility, API restrictions and permitted usage. Confirm the intended independent accounts/projects yourself; different key strings and labels do not prove independence. Do not change accounts or projects to evade service restrictions.
4. Run `python scripts/check_gemini.py` for safe configuration validation. `--catalogue --slot 1` checks configured model capabilities without sending documents. Repeat explicitly for the other configured slots. `--live --slot 1` optionally generates two sentences about evaporation. These network paths never run during normal tests. No credentials means an explicit skipped result.
5. Run `python prepare_models.py` and optionally `python prepare_models.py --models summarizer`, then `python app.py`. Restart after changing keys or settings. Missing Gemini keys do not prevent startup or local QA.

Project/account metadata is declared, not independently verified by this application. Supplying a project ID does **not** alter an API key's authorization/quota scope. Catalogue access verifies model visibility, not project ownership, account independence, billing capacity or service reliability.

`.env`, `.env.*` (except placeholder examples), local environments, model weights and private key files are already excluded by `.gitignore`. Credential dataclasses suppress secret representation. Provider exception strings, headers, prompts and documents are never logged. Unexpected AI failures produce only a fixed `internal_error` log message. Known configured secrets in provider output are rejected. Only selected original evidence reaches Gemini; documents are not uploaded to Google Files. No tools, code execution, URL retrieval or automatic function calling are enabled.

## Verified models and generation settings

Official documentation checked on 23 October 2026:

| Preference | Stable model ID | Default thinking | Supported thinking levels |
|---|---|---|---|
| Primary | `gemini-3.8-flash` | `low` | low, medium, high |
| Secondary | `gemini-3.6-flash` | `minimal` | minimal, low, medium, high |
| Lightweight fallback | `gemini-3.5-flash-lite` | `minimal` | minimal, low, medium, high |

Sources: [3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash), [3.6 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash), [3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite), [thinking capabilities](https://ai.google.dev/gemini-api/docs/thinking), [official Python SDK](https://googleapis.github.io/python-genai/), [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits). Account access and deployed availability still require the optional live check. All three support text generation and thinking; 3.8 rejects `minimal`. The installed SDK types were exercised without network, including `GenerateContentConfig`, `ThinkingConfig`, per-call `HttpOptions` and `HttpRetryOptions(attempts=1)`.

`GEMINI_MODELS` reorders/subsets the verified registry; `GEMINI_MODELS_1` through `_3` restrict each slot's model eligibility. Unverified/duplicate model IDs fail configuration validation rather than silently substituting. To add a future model, first verify and extend the registry with tested capabilities. SDK uses the documented `generate_content` endpoint, `v1beta`, with no preview model IDs.

| Setting | Default | Meaning |
|---|---:|---|
| `QNEXT_AI_ENABLED` | true | Server-wide generation switch |
| `GEMINI_ATTEMPT_TIMEOUT` | 8 seconds | Maximum wait per provider attempt |
| `GEMINI_DEADLINE` | 20 seconds | Total gateway request deadline |
| `GEMINI_MAX_ATTEMPTS` | 4 | All keys/models combined |
| `GEMINI_BACKOFF` | 0.4 seconds | Exponential delay with up to 25% jitter |
| `GEMINI_COOLDOWN` | 30 seconds | Transient circuit cooldown |
| `GEMINI_AUTH_COOLDOWN` | 300 seconds | Auth/permission/model configuration cooldown |
| `GEMINI_CONTEXT_CHARS` | 10000 | Total selected evidence characters |
| `GEMINI_MAX_PASSAGES` | 6 | Selected passage cap |
| `GEMINI_MAX_OUTPUT_TOKENS` | 1800 | Maximum adaptive token budget |
| `GEMINI_TEMPERATURE` | 0.2 | Configurable sampling variance; no factual guarantee |
| `GEMINI_THINKING_LEVEL` | auto | Model-specific minimum supported level |

Malformed/nonfinite/out-of-range settings safely disable the AI path and leave retrieval functional. Simple facts receive up to 500 output tokens; definitions 800; mechanisms/concepts 1200; comparisons/multipart/complex tasks 1800, bounded by the configured maximum. Topic structure, not question length alone, drives selection.

The evidence builder prioritizes the existing result's raw source passages, IDs and offsets, adds relevant TF-IDF passages when some evidence exists, and gives selected sources fair space under the cap. It retains original answer/status/reason/verification metadata. Omitted passages and clipped excerpts are flagged. Missing-answer requests carry no fabricated supporting passages. Unavailable verification and below-threshold results remain uncertain. Prompts explicitly protect numeric values, units, dates and negation, separate illustrative examples/outside knowledge, and treat document instructions as untrusted data. Prompt instructions cannot prove factual correctness or perfectly prevent injection; the application enforces data isolation, safe rendering and lack of operational tools separately.

Only one completed `STOP` candidate with nonempty text is accepted. Safety blocks/refusals stop recovery. `MAX_TOKENS` is incomplete, not success. Empty/malformed results may use eligible bounded recovery. Unknown `[P…]` citations are rejected; actual citations map to selected passage IDs and excerpt offsets. Citation validation proves reference existence, not that each generated claim follows from that reference. Semantic correctness, contradictions not detected by local NLP, and unmarked outside knowledge still require human review.

## Failover and quota handling

Selection is model-first, then eligible credential. Invalid primary credentials therefore permit another healthy slot to use the same preferred model. One generation runs at a time per slot; there are at most three provider worker threads. No speculative competing generations are made.

| Failure | Action |
|---|---|
| 401 / structured invalid API key | Cool affected slot; try another healthy slot |
| 402/403 | Cool affected project; structured account restriction cools declared account |
| Structured model-only 429 | Cool that project/model pair; permit another eligible project or model |
| Daily/zero quota or unknown-scope 429 | Cool whole project; never rotate keys sharing it |
| Unknown project ownership | Never assume independence; conservatively share cooldown with potentially related slots |
| 503/5xx | Temporarily cool model and try a verified fallback |
| 404 | Cool unavailable model and try verified fallback; no identical retries through all keys |
| Permanent 400/422 | Terminal request/configuration failure |
| Timeout | Cool the affected model; try a different eligible model within the remaining shared budget; never resend the same model request |
| Safety/refusal | Terminal; never rotate accounts to bypass the refusal |
| Empty/incomplete output | Pair cooldown; eligible bounded recovery |

Structured `QuotaFailure`, `RetryInfo`, `ErrorInfo` and `Retry-After` are parsed from actual SDK exceptions. Unknown rate-limit scope is treated conservatively as project-wide. Retry hints delay the affected scope; independent healthy scopes may proceed. If no eligible route can recover before the deadline, return promptly. Small transient backoff uses jitter; repeated failures open a longer cooldown. No network request happens after the total attempt budget. Under cooldown, one slot lease allows one recovery probe when eligible. A previous in-flight success cannot erase a newer failure.

SDK automatic retries are disabled. Per-call network timeouts and bounded future waits enforce the caller's wall-clock budget even if a transport stalls. A timed-out worker keeps its slot lease until the actual call ends; unrelated requests cannot spawn unbounded replacements. Recovery uses a different configured model and an available slot. With only one slot, the gateway waits for its own earlier transport to finish before reusing the slot, within the same deadline. An ongoing remote generation cannot be recalled and may have consumed quota; fallback can consume additional quota, bounded by the shared attempt/deadline limits. Late primary output is discarded and its terminal HTTP status is logged without content. There is no public browser retry loop.

Health and clients are shared only inside one process under locks. Dataset operations remain serialized within each session, not across unrelated sessions. Multiple WSGI processes do not coordinate quota state or documents; use the application's documented single-process deployment. Real capacity depends on each project's/account's service limits, not the number of configured keys.

## Metrics and verification

`GET /api/ai/status` exposes only safe readiness information. `GET /api/ai/stats` returns metrics for the current session: enabled/disabled requests, successes/unavailable results, attempts/failovers, no-document-answer requests, selected models/slot numbers, and fixed failure-category counts (including 429/quota, server, timeout, empty/incomplete output). p50 uses the latest 200 completed AI calls; p95 is null until at least 20 samples. Failures remain in that latency window, so these values are operational timings, not success-only provider benchmarks. Original `/api/stats` remains retrieval-only. Metrics contain no prompt, evidence, account email or key.

Commands:

```powershell
.venv\Scripts\python.exe -m pytest tests -q --junitxml=docs/verification/final-python.xml
node --test tests/js/test_ai_panel.mjs tests/js/test_report.mjs tests/js/test_speech_input.mjs
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe scripts/evaluate_ai.py --output docs/verification/offline-evaluation.json
.venv\Scripts\python.exe scripts/check_gemini.py --live
# Explicit opt-in, synthetic questions only; may incur charges:
.venv\Scripts\python.exe scripts/evaluate_ai.py --live --all-models --output docs/verification/live-evaluation.json
```

Baseline under Python 3.12 without local neural dependencies/weights: **102 passed, 11 failed**. System Python 3.13 initially failed collection with missing Flask. The 11 baseline failures involved missing semantic/summarizer resources or neural-dependent expected spans/answers. Pinned local dependencies/models were installed/prepared; original assertions were preserved. The summarizer preparation defect was repaired as described above. The final test result and full logs are in `verification/final-python.txt` / `.xml`. Original baseline output is in `verification/baseline-python.txt` / `.xml`. Normal CI uses mocked Gemini and does not require keys; legacy neural regression tests still require their prepared local models.

Final verification: **204 Python tests passed, zero failed/skipped**, comprising all **113 original tests** and **91 new tests**, in **100.30 seconds**. Three upstream tokenizer future warnings remain. The complete model-preparation command also succeeded after the summarizer repair, and `pip check` reported no broken requirements. Syntax checks passed for both frontend scripts. The final preservation scan found **35 original files unchanged**, eight modified originals, no Google API-key patterns and no real `.env` file created; see `verification/change-audit.json` and `verification/results.json`. This folder has no Git repository, so no commit/push or Git diff was possible.

The initial frontend baseline was **21 passed**; new UI tests extend that suite to **27 passed**, zero failed/skipped. They cover boolean toggle transmission, no call while disabled, retrieved output before generation, preserved retrieval on failure, loading/uncertainty/general-knowledge labels, safe source/model rendering and actual page wiring. Report and speech tests remain unchanged. Browser verification used the real sample-loading and ask flows against loopback, confirming Retrieved Text remains visible alongside the unavailable-AI panel with no credentials. A second browser question with AI off returned the original no-answer message and hid the AI panel. The temporary verification servers were stopped. Visual evidence: `verification/browser-ai-unavailable.png`.

## Quality rubric and measured limits

The fixed synthetic set contains 16 cases covering direct facts, definitions, concepts, steps, comparisons, multiple passages, paraphrases, absent/ambiguous/conflicting evidence, numeric/date/negation details, prompt injection, long context, empty output and truncation. Reference responses are authored examples, not claimed Gemini generations. Offline evaluation measures **contracts and local orchestration only**. `offline-evaluation.json` includes actual timings and explicitly records `provider_latency_measured:false`; those millisecond numbers do not establish the 3-second median or 8-second p95 provider targets.

For a live evaluation, use the same questions across each accessible model, retain generated text and status, and review each output alongside its synthetic sources. Score each dimension 0–2:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Correctness/source meaning | Wrong relation, number or negation | Partly correct, material omission | Correct required facts and constraints |
| Grounding/uncertainty | Unsupported source claim or false certainty | Weak attribution/qualification | Accurate citations and explicit missing/conflicting status |
| Completeness/relevance | Fails the question | Covers some parts or repeats | Covers all requested parts directly |
| Readability/length | Confusing, incomplete or excessively long | Understandable with avoidable filler | Clear student-level explanation of suitable length |

Any fabricated fact/source, changed unit/negation or concealed conflict is a critical failure regardless of aggregate score. Required-fact substring screens are review aids, not semantic correctness proofs. No LLM judge is used. Compare success rate, terminal failure categories, output tokens where available and complete-request p50/p95 with sample size, OS/Python/hardware and model configuration stated. Use at least 20 samples per model before interpreting p95, and repeat under a defined workload. Never infer provider performance from mocks.

No usable Gemini credentials were configured during the original verification above, so that live smoke test returned **skipped**. See the subsequent configured-credential verification below. Broad educational quality and inter-model quality comparisons still require the documented evaluation.

## Troubleshooting

- **Not configured:** add at least one real server-side key. Check `.env` location, then restart. Retrieval needs no Gemini configuration.
- **Configuration error:** `scripts/check_gemini.py` lists fixed, secret-free validation messages. `minimal` is invalid if 3.8 Flash is configured; use `auto` or `low`.
- **Quota / permission / billing:** inspect the actual owning project/account in Google's console. Respect its reset time and eligibility. Duplicate keys in one project do not restore quota.
- **Circuit open:** wait for the scoped cooldown; expired routes permit a controlled probe. Restart only after an operator corrects configuration, not as an automated quota workaround.
- **Timeout:** local retrieval is still usable. A different model may recover within the shared budget; the timed-out model is not resubmitted. Inspect structured `qnext.ai` logs using the returned opaque `diagnostic_id`. Do not paste keys or document text into logs.
- **Model unavailable:** run the explicit catalogue check. Change only to documented, tested registry entries.
- **Local semantic/summarizer unavailable:** install the pinned local dependencies and run `prepare_models.py`; this is separate from Gemini. The default optional BART legacy state dict is converted with `weights_only=True`; no arbitrary pickle loader is used by preparation.
- **Session or document replaced:** ask again against the active document. Deferred requests deliberately reject old IDs, including same-text replacements.

This is a tested local integration, not a certification of production deployment or perfect generated factuality. Existing limitations remain: one ephemeral process, no OCR, no conversation-memory reasoning, no universal contradiction detector, and retrieval-only browser export/history reload constraints.

## Configured-credential follow-up — 23 October 2026

After the operator supplied credentials, all three slots passed local configuration validation with no errors/warnings. Read-only Google catalogue checks confirmed `generateContent` support for all three configured model IDs through each of the three slots. These checks confirm access, not independently verified project ownership or available quota capacity.

Two integration defects were corrected without changing the saved credentials:

- The key validator now accepts Google's newer `AQ.` authorization-key family while still rejecting unexpected punctuation, whitespace and control characters. See [Google's auth-key documentation](https://ai.google.dev/gemini-api/docs/api-key).
- Google rejected the previous SDK network deadline with `Minimum allowed deadline is 10s`. The transport respects the model registry's 10-second server minimum. The gateway's independent future enforces the configured caller wait (8 seconds by default) and total request deadline (20 seconds). At this earlier checkpoint, timeouts were terminal; the subsequent investigation below corrected that recovery limitation.

Affected deterministic tests: **95 passed**, zero failures, using `python -m pytest tests/test_gemini_gateway.py tests/test_ai_explainer.py tests/test_ai_integration.py tests/test_ai_evaluation.py -q`. Four new key-format cases supplement the earlier 204-test full-suite result. `pip check` reported no broken requirements. Follow-up test logs are in `verification/auth-key-followup-tests.txt` and `.xml`.

The preferred `gemini-3.8-flash` subsequently produced an 8-second caller timeout and, on one separately bounded diagnostic request, a temporary server error. One harmless synthetic question using the configured fallback `gemini-3.6-flash` completed successfully through slot 1 in **3234 ms**. This is one successful smoke response, not a median/p95 benchmark or educational-quality evaluation. Model order and the operator's saved timeout preferences remain unchanged. No document content was sent, and no credentials were printed or copied into diagnostic artifacts.

## Failure recovery and switch refinement

The subsequent [failure investigation report](GEMINI_FAILURE_TOGGLE_REPORT.md) records the reproduced primary server failure/secondary timeout, the terminal-timeout recovery defect, structured diagnostics, updated prompt guidance, and the compact animated switch. The original helper sentence and its markup/status-fetch code were removed. The native switch uses the server's configured default, retains checked/disabled semantics and keyboard focus, and respects reduced motion. Current verification: **218 Python tests** and **29 JavaScript tests** passed. Synthetic end-to-end generation recovered after a primary timeout and returned a document-grounded explanation through the secondary model in **10342 ms total**.
