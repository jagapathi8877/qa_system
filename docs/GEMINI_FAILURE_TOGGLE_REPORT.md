# Gemini failure investigation and AI switch refinement

Verified on 23 October 2026, in the existing Flask/vanilla-JavaScript checkout. No Git metadata was available. Pre-change copies of affected source files are retained under `verification/toggle-failure-before/`; environment files and credentials are excluded.

## Diagnosis and evidence

The original screenshot alone cannot establish an upstream cause. The old gateway kept failure categories in API results but did not log provider attempt/status context. Its original screenshot request's exact provider response was therefore not recoverable from the existing terminal logs.

A fresh reproduction against the running port-5000 application used a new cookie/session and harmless synthetic cross-validation text. It returned `unavailable` with `failures: [server_error, timeout]`, model `gemini-3.6-flash`, slot 1, two attempts and 13336 ms AI latency; retrieval was preserved. See `verification/toggle-failure-reproduction.json`.

The code defect was an unconditional terminal return after a timeout in `GeminiGateway.generate`. This bypassed the remaining model candidates, including Flash-Lite, despite time/attempt budget remaining. The primary also timed out on fresh isolated checks. Newly added late-completion diagnostics recorded the preferred model's eventual **HTTP 504**. Authentication/model catalogue access succeeded, so there is no evidence that these failures were caused by invalid keys or nonexistent model IDs.

After the repair, a synthetic request through `/api/ask` → `/api/explain` returned:

- Original retrieval: `answered`, with the answer unchanged in its independent retrieval snapshot.
- First attempt: `gemini-3.8-flash`, slot 1, caller timeout at 8000 ms; late completion was HTTP 504 at 11281 ms.
- Second attempt: `gemini-3.6-flash`, slot 2, HTTP 200 at 2328 ms.
- Final AI result: `success`, `document_evidence`, 707 characters, 10342 ms total including context construction and recovery.

See `verification/toggle-live-http.json` and `verification/toggle-attempt-diagnostics.json`. A separate smoke call succeeded after timeout recovery in 10359 ms total. A browser suggestion question also returned a generated explanation and traceable source IDs after recovering through the secondary model. These are individual live observations, not median/p95 or general educational-quality measurements.

## Changes

| Files | Change |
|---|---|
| `gemini_gateway.py` | Recover timeouts through a different configured model, retain the original slot lease, discard late text, log terminal late status; keep the total 20-second/four-attempt defaults. Add secret-free JSON attempt/request diagnostics and explicit unsupported-parameter classification. |
| `app.py` | Enable the dedicated AI log stream; classify/log unexpected explanation-boundary failures without exception text; pass configured AI default into the template. |
| `ai_explainer.py` | Ask for explanations beyond source repetition, with adaptable concision guidance; retain original evidence/uncertainty rules and log rejected citations without content. |
| `templates/index.html`, `static/style.css` | Replace the checkbox presentation with a compact sliding native switch, restrained purple gradient, neutral OFF track, ON/OFF label, keyboard focus and reduced-motion transitions. Remove the helper paragraph and its description reference. |
| `static/ai-panel.js`, `static/script.js` | Bind checked state to submission state and ON/OFF text; remove the helper-only configuration fetch. Existing typed/suggestion flows still share the same request path. |
| `tests/test_gemini_gateway.py`, `tests/test_ai_integration.py`, `tests/js/test_ai_panel.mjs` | Cover timeout recovery, single-slot lease recovery, two failed models followed by Flash-Lite, all credentials unavailable, safe structured diagnostics, unsupported parameters, configured switch defaults, helper absence and native state changes. |
| `docs/AI_INTEGRATION.md`, this report, `docs/verification/toggle-*` | Current recovery policy, reproducible evidence and limitations. |

No keys, project declarations, account labels, model order, timeout preferences, local retrieval modules, original Python tests, speech module or report generator were changed.

## Request path and recovery policy

The switch is a checkbox with `role=switch`, an accessible name and native checked/disabled state. The rendered default follows `QNEXT_AI_ENABLED`; ordinary configuration remains ON. `bindToggle` synchronizes the ON/OFF label and submission boolean without touching corpus/session state. Typed submissions and suggestion chips both call the existing `askQuestion` flow.

The browser sends JSON booleans `ai_mode` and `defer_ai`. `/api/ask` runs the original local QA pipeline and returns its unchanged result immediately. `/api/explain` accepts only a previously authorized question from that session and active document revision. AI OFF skips the explainer and all Google calls. The two result panels remain separate; source text and generated output are rendered as safe text nodes.

The explainer sends the question, retrieval status/reason, original answer, selected original evidence/offsets and uncertainty metadata. It uses adaptive answer budgets and asks for explanation of mechanisms, relationships and examples where useful. Missing evidence remains labelled general knowledge; partial/conflicting evidence remains qualified. Empty/malformed/truncated output is unsuccessful. Citation IDs must refer to selected evidence. These controls do not prove every generated claim correct; human review remains necessary.

Recovery remains error-specific:

- Invalid credentials cool only their slot; another healthy slot can use the preferred model.
- Project quota exhaustion cools the project and potentially related unknown ownership groups; another key in that same project is not treated as independent capacity.
- Model outage/not-found cools the affected model and allows a verified fallback.
- Invalid/unsupported request parameters terminate with a fixed reason code, rather than rotating keys.
- Billing/permission and declared account restrictions use their corresponding project/account scope.
- Safety/refusal remains terminal; no account rotation bypasses it.
- Timeout excludes the same model for the rest of that request. Its worker retains the slot until the actual transport ends. An eligible different model can recover within the same global attempt/deadline budget. If only one slot exists, reuse waits for that request's own transport to end; unrelated requests do not queue behind another user's generation.
- Exhausted budgets or no eligible route return safe unavailability while retaining retrieval.

Timeout fallback can consume extra provider quota because an earlier remote generation may already be running. It is strictly bounded and does not resubmit the same uncertain model request. No speculative routine generations, new projects, new credentials or quota workarounds were introduced.

## Diagnostics and configuration verification

The `qnext.ai` stream emits JSON with event, opaque diagnostic ID, model ID, slot number, attempt, fixed category/reason, scope, HTTP status where known and elapsed milliseconds. Project groups are hashed, and account identifiers are omitted. Exception strings, keys, authorization headers, prompts and document contents are never logged. Late outcomes are recorded separately and cannot overwrite the returned explanation. The frontend receives safe category/message fields and an opaque diagnostic ID; it never receives provider exception text.

Environment slots remain `GEMINI_API_KEY_1..3`, `GEMINI_PROJECT_ID_1..3`, and `GEMINI_ACCOUNT_LABEL_1..3`. Fresh configuration checks loaded three slots without errors/warnings. Nine read-only catalogue checks (three models through three slots) confirmed generation support. These do not independently verify declared ownership, account independence or quota capacity. Existing zero/one/three-slot and shared-project mocked tests remain intact.

Official documentation rechecked for the unchanged order: [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash), [Gemini 3.6 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash), [Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite), [thinking](https://ai.google.dev/gemini-api/docs/thinking). All are documented stable models and accessible through the actual configured keys. The official installed SDK uses `generate_content`, a key-specific client, no automatic SDK retries, per-model supported thinking (`low` for 3.8; `minimal` for the others), and the provider's previously verified minimum network deadline. Caller/deadline bounds are enforced separately.

## Tests and browser verification

Baseline: 95 existing AI tests and 27 JavaScript tests passed before this change.

```powershell
.venv\Scripts\python.exe -m pytest tests -q --junitxml=docs/verification/toggle-full-python.xml
.venv\Scripts\python.exe -m pytest tests/test_gemini_gateway.py tests/test_ai_explainer.py tests/test_ai_integration.py tests/test_ai_evaluation.py -q --junitxml=docs/verification/toggle-ai-tests.xml
node --test tests/js/test_ai_panel.mjs tests/js/test_report.mjs tests/js/test_speech_input.mjs
node --check static/script.js
node --check static/ai-panel.js
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe scripts/check_gemini.py --catalogue
.venv\Scripts\python.exe scripts/check_gemini.py --live
```

Full Python regression result: **218 passed, zero failed/skipped, 85.79 seconds**, including all original neural-model tests. Three pre-existing upstream tokenizer future warnings remain. The affected AI subset passed **105 tests**. JavaScript passed **29 tests**, including all original speech/report tests. Syntax checks and dependency validation passed. Ordinary tests use mocked Gemini and do not require live credentials.

Actual browser checks used a separate sample-data session: ON by default; suggestion submission displayed retrieved evidence immediately and later a generated explanation; Space switched OFF with visible keyboard focus while preserving the document and retrieved answer; a typed OFF question still answered through retrieval with no AI panel. At a 390×844 mobile viewport, document width was 375 px (no horizontal overflow), the OFF label and compact switch remained visible, and the removed sentence was absent. The temporary viewport was reset after verification.

Visual artifacts: `verification/premium-toggle-on.png`, `verification/premium-toggle-off.png`, `verification/premium-toggle-mobile.png`. Security check: `verification/toggle-security.json` found zero configured-key occurrences in deliverable sources/logs, confirmed helper absence and byte-identical core NLP/speech/report modules. This checkout has no Git metadata, so review used pre-change snapshots and source hashes.

The original helper sentence is absent from the rendered template, JavaScript and CSS. Its helper element, description reference, helper-only CSS and status-fetch JavaScript were removed; it was not replaced with another explanatory paragraph.

The verified build was restarted at `http://127.0.0.1:5000`. A final separate-session check on that main address confirmed HTTP 200, both local models loaded, three configured slots, premium-switch markup, helper absence, AI OFF with **zero provider requests**, and identical corpus identity/retrieved answer across the OFF/ON requests. Live ON generation returned `success`/`document_evidence` with 593 usable characters in **8976 ms total**, recovering from a primary HTTP 503 through a secondary HTTP 200 on slot 1. See `verification/toggle-main-app.json` and `verification/toggle-main-attempts.json`. The temporary verification server was stopped; the main application remains running.

The app remains single-process and ephemeral. A server restart requires reloading the uploaded document; toggling AI itself does not clear it. Provider availability/latency still vary. The observed primary timeouts/504s are external service outcomes; the repaired application now recovers through eligible configured models when its budget allows and preserves retrieval if recovery cannot succeed.
