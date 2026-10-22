"""Explicit one-time model download; existing pinned Python packages suffice.

python prepare_models.py [--models qa,embedder,reranker]
Uses a copy-based local directory on Windows, with pinned commit metadata.
"""
import argparse,hashlib,json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
import models

PINNED_REVISIONS={
 'deepset/minilm-uncased-squad2':'934656cdda79824eabf503ed56e15c01ddbdbe3f',
 'sentence-transformers/all-MiniLM-L6-v2':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
 'cross-encoder/ms-marco-MiniLM-L6-v2':'233902d25c440f23af6f7d6e94d2946bac0bee0a',
 'cross-encoder/nli-MiniLM2-L6-H768':'b95119ce93d3e065de6214e38cd4a97b0f2f2c6d',
}

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',default='qa,embedder,reranker,verifier');args=p.parse_args()
    names={'qa':models.QA_MODEL,'embedder':models.EMBEDDER_MODEL,'reranker':models.RERANKER_MODEL,'verifier':models.VERIFIER_MODEL,'summarizer':models.SUMMARIZER_MODEL}
    existing=models.MODEL_CACHE/'preparation.json'
    manifest=json.loads(existing.read_text()) if existing.exists() else {}
    for role in args.models.split(','):
        name=names[role];revision=PINNED_REVISIONS.get(name)
        if revision is None:revision=HfApi().model_info(name).sha
        target=models.MODEL_CACHE/name.replace('/','--')
        snapshot_download(name,revision=revision,local_dir=str(target),allow_patterns=['*.json','*.txt','*.safetensors','*.model'],ignore_patterns=['onnx/*','openvino/*','*.onnx'],max_workers=2)
        if not (target/'model.safetensors').exists():
            raise RuntimeError('Safe model weights are missing for '+name)
        files={str(f.relative_to(target)):hashlib.sha256(f.read_bytes()).hexdigest() for f in target.rglob('*') if f.is_file() and '.cache' not in f.parts and f.name!='quillqa_manifest.json'}
        record={'model_id':name,'revision':revision,'files':files}
        (target/'quillqa_manifest.json').write_text(json.dumps(record,indent=2))
        manifest[role]=record
        print('Prepared',role,name,revision,flush=True)
    (models.MODEL_CACHE/'preparation.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
