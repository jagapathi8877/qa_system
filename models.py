"""Model identities and explicit, offline-first local preparation paths.

No remote code or network requests run during inference. Run prepare_models.py
once to populate .models/, or configure the corresponding environment variable.
"""
import os
from pathlib import Path

MODEL_CACHE = Path(os.environ.get('QUILLQA_MODEL_CACHE', str(Path(__file__).resolve().parent / '.models')))
QA_MODEL = os.environ.get('QUILLQA_QA_MODEL', 'deepset/minilm-uncased-squad2')
EMBEDDER_MODEL = os.environ.get('QUILLQA_EMBEDDER_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
RERANKER_MODEL = os.environ.get('QUILLQA_RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L6-v2')
VERIFIER_MODEL = os.environ.get('QUILLQA_VERIFIER_MODEL', 'cross-encoder/nli-MiniLM2-L6-H768')
SUMMARIZER_MODEL = os.environ.get('QUILLQA_SUMMARIZER_MODEL', 'sshleifer/distilbart-cnn-6-6')


def local_path(name: str) -> str:
    configured = Path(name)
    if configured.is_dir():
        return str(configured.resolve())
    prepared = MODEL_CACHE / name.replace('/', '--')
    return str(prepared.resolve()) if (prepared / 'config.json').is_file() else name
