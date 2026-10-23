# QNext document question answering

Load one TXT, PDF, DOCX, PPTX, CSV or JSON document, or paste text, and ask questions grounded in that document. A replacement upload replaces the active corpus. Scanned PDFs need an external OCR step.

The backend retrieves passages, ranks question–passage and question–clause relevance, uses a SQuAD2 reader that can decline to answer, and checks supported question forms against source clauses or textual entailment. Dates, polarity, units, qualifiers and conflicting candidate answers receive separate checks. Answers include exact source spans and offsets. This improves meaning matching but cannot guarantee understanding every question or verifying real-world truth.

## Run locally on Windows

Use Python 3.10–3.12 and the pinned dependencies. Verification used Python 3.12.14. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe prepare_models.py
.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000. Preparation needs internet access and substantial disk space. Four default reader/retrieval/verifier revisions are pinned; preparation records commit IDs and SHA256 hashes in `.models/preparation.json`. Models are ignored generated artifacts. This checkout has a tested Python 3.12.14 environment at `.venv`.

## Optional Gemini explanations

The local retrieval engine still produces the original **Retrieved Text**, evidence and no-answer result. A separate **AI-Generated Explanation** explains selected evidence, labels general knowledge, and communicates incomplete or conflicting evidence. The browser defaults AI on; turning it off prevents generation on the backend. Missing credentials and Gemini failures leave retrieval available. Questions and selected original passages are sent to Google only for explicitly AI-enabled requests.

Copy `.env.example` to `.env` and fill `GEMINI_API_KEY_1` / `GEMINI_PROJECT_ID_1` / `GEMINI_ACCOUNT_LABEL_1`. Slots 2 and 3 support independently authorized projects/accounts; one slot also works. Use opaque account labels, not emails. Restart after changing configuration. The server does not verify ownership from a declared project ID; operators must check the actual key/project/account association in Google AI Studio and Cloud Console.

Stable defaults, verified in Google's documentation on 23 October 2026: `gemini-3.8-flash`, `gemini-3.6-flash`, then `gemini-3.5-flash-lite`. Generation uses the official `google-genai==2.33.0` SDK. [Full configuration, quota policy, API, tests, evaluation rubric and troubleshooting](docs/AI_INTEGRATION.md).

```powershell
Copy-Item .env.example .env
# Edit .env locally; never paste keys into frontend files or commit them.
.venv\Scripts\python.exe scripts/check_gemini.py
# Explicit, optional network checks; catalogue sends no document content:
.venv\Scripts\python.exe scripts/check_gemini.py --catalogue
.venv\Scripts\python.exe scripts/check_gemini.py --live --slot 1
.venv\Scripts\python.exe app.py
```

Executive summaries additionally need:

```powershell
.venv\Scripts\python.exe prepare_models.py --models summarizer
```

Inference reads prepared models or an existing Hugging Face cache without network access. No automatic NLTK download is required; `QUILLQA_DOWNLOAD_NLTK=1` explicitly enables resource preparation. `/api/health` reports loaded models. Missing QA uses conservative relation matching and frequently abstains. Semantic mode needs embeddings. Missing summarizer returns HTTP 503. Failed model loads retry after 30 seconds; preparing the reader can also trigger recovery without restart.

## Answer and summary contracts

- Retrieval scores measure similarity to the returned source passage. TF-IDF and semantic scores have different scales. An optional threshold must be finite, between 0 and 0.9, and applies to the selected retrieval mode.
- `confidence` is an uncalibrated reader span score, or null for deterministic/source-copy paths. `confidence_type`, `confidence_calibrated`, `answer_method`, `meaning_check` and `sources` explain the evidence. Entailment scores estimate textual support, not the probability of factual truth.
- Unsupported questions, unresolved pronouns, ambiguity and conflicting candidate measurements return `no_answer`. Explicit negative evidence can correct a false premise. Multipart/comparison and relationship-following use a limited grammar; arbitrary reasoning, arithmetic, conversation memory and universal conflict detection are not supported.
- Latest/current requests refer to the latest dated supporting record in the dataset, rather than live current knowledge.
- Summaries and explanations retain source sentences. Generated paraphrases are accepted only when they exactly align to source sentences; otherwise extractive selection supplies the output. This favors factual fidelity over free rewriting. Executive summaries sample at most the first 24 passages. Coverage and warnings survive cache hits. Whole-document extractive analysis samples at most 500 sentences across the document and reports that limit.

## API and limits

`POST /api/paste` takes `{text}`; `POST /api/upload` takes multipart `file`; `POST /api/ask` takes `{question, retriever?, threshold?, ai_mode?, defer_ai?}`. Legacy requests that omit `ai_mode` remain retrieval-only. The original fields are unchanged; `retrieval`, `ai` and `request_id` are additive. The UI uses deferred generation through `POST /api/explain` so evidence appears before AI completes. Other endpoints: `POST /api/sample`, `GET /api/dataset`, `DELETE /api/dataset`, `GET /api/suggest`, `POST /api/summarize`, `GET /api/stats`, `GET /api/history`, `GET /api/health`, `GET /api/ai/status`, `GET /api/ai/stats`.

Invalid JSON schemas, question types, empty questions, modes and thresholds return JSON 400 before counters change. Parsing/content failures use 422, request-size failures 413. Limits: 2,000,000 text characters, 5,000 bounded passages, 1,000 question characters, 25 MiB request bodies, 100 MiB expanded Office archives. Malformed CSV quoting, surplus cells, duplicate/empty headers are rejected. Nested JSON preserves paths; DOCX preserves body paragraph/table ordering. PDF/PPT layout and unusual encodings remain format-specific limitations.

State and caches are **ephemeral, single-process memory**. Use one process; independent workers do not share datasets. Restart loses state. There are at most 50 loaded sessions. Idle entries can expire after one hour when admitting another corpus. Capacity returns 503 rather than silently evicting an active dataset. Session operations are serialized; CPU inference is bounded. These controls are not a universal memory budget or public deployment architecture.

`/api/history` keeps the latest 1,000 Q&A records with corpus identity and separate AI output. The frontend does not restore the answer cards on reload. PDF exports read this server session history, including questions from before a page refresh, and present numbered questions, retrieved text and AI explanations in a plain black-on-white portrait report. Long answers continue across pages; disabled or unavailable AI is labelled. Resetting the session or restarting the server clears this history. CSV retains its existing retrieval-only schema and current-page log. Source passages are expandable in the answer card; retriever/confidence fields also exist in the API. Browser speech recognition may use the browser vendor's service.

Debug defaults off. Development generates a random signing secret per process. `QUILLQA_ENV=production` requires a private `QUILLQA_SECRET` of at least 32 characters. Persistent shared storage, authentication, TLS deployment and frontend history/export integration remain separate work. This is not a production-readiness claim.

Model configuration: `QUILLQA_QA_MODEL`, `QUILLQA_EMBEDDER_MODEL`, `QUILLQA_RERANKER_MODEL`, `QUILLQA_VERIFIER_MODEL`, `QUILLQA_SUMMARIZER_MODEL`, `QUILLQA_MODEL_CACHE`. Custom models need compatible architectures and verifier labels in contradiction/entailment/neutral order. `QUILLQA_CPU_THREADS` controls inference threads (default two).

## Verification

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q
node --test tests/js/test_speech_input.mjs tests/js/test_report.mjs tests/js/test_ai_panel.mjs
.venv\Scripts\python.exe scripts/evaluate_ai.py --output docs/verification/offline-evaluation.json
```

Real-model tests need prepared models, including the optional summarizer. `-m "not slow"` skips marked integrations but some historical tests still call the reader. The [implementation report](docs/IMPLEMENTATION_REPORT.md) and [file-level change record](docs/CHANGELOG.md) describe the verified changes and remaining limitations. In the original local workspace, parent folders `audit/` and `phase2/2026-10-22-quality/` retain baselines, frozen question sets and raw responses; those local artifacts are excluded from this source repository. Small synthetic regression scores are not general benchmark accuracy.

## Model attribution

Reader: [deepset MiniLM SQuAD2](https://huggingface.co/deepset/minilm-uncased-squad2), CC BY 4.0. Embeddings: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache 2.0. Ranking: [MS MARCO MiniLM](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), Apache 2.0. Textual inference: [NLI MiniLM2](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768), Apache 2.0. Optional summarizer: [DistilBART CNN](https://huggingface.co/sshleifer/distilbart-cnn-6-6), Apache 2.0. Consult model cards and bundled licenses before redistribution. Weights are not committed with source.
