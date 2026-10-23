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
            if role != 'summarizer' or name != 'sshleifer/distilbart-cnn-6-6':
                raise RuntimeError('Safe model weights are missing for '+name)
            # This upstream summarizer publishes only a legacy state dict.
            # Read tensors with torch's restricted loader, never arbitrary
            # pickle execution, then save safe weights for offline inference.
            import torch
            from transformers import AutoConfig, AutoModelForSeq2SeqLM
            snapshot_download(name,revision=revision,local_dir=str(target),
                              allow_patterns=['pytorch_model.bin'],max_workers=1)
            weights=torch.load(target/'pytorch_model.bin',map_location='cpu',weights_only=True)
            if not isinstance(weights,dict) or any(not isinstance(k,str) or not isinstance(v,torch.Tensor) for k,v in weights.items()):
                raise RuntimeError('The summarizer artifact is not a tensor state dictionary.')
            model=AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(str(target),local_files_only=True))
            if ('lm_head.weight' not in weights and 'model.shared.weight' in weights
                    and model.config.tie_word_embeddings
                    and model.lm_head.weight.data_ptr()==model.model.shared.weight.data_ptr()):
                weights['lm_head.weight']=weights['model.shared.weight']
            model.load_state_dict(weights,strict=True)
            model.save_pretrained(str(target),safe_serialization=True)
        files={str(f.relative_to(target)):hashlib.sha256(f.read_bytes()).hexdigest() for f in target.rglob('*') if f.is_file() and '.cache' not in f.parts and f.name!='quillqa_manifest.json'}
        record={'model_id':name,'revision':revision,'files':files}
        (target/'quillqa_manifest.json').write_text(json.dumps(record,indent=2))
        manifest[role]=record
        print('Prepared',role,name,revision,flush=True)
    (models.MODEL_CACHE/'preparation.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
