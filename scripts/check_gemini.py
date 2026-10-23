"""Explicit credential/catalogue validation and harmless live smoke test.

No network request without --catalogue or --live. Never prints provider errors,
project/account metadata, prompts, keys, or raw catalogue responses.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env', override=False)
from ai_config import AIConfig
from gemini_gateway import GeminiGateway, classify_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--catalogue', action='store_true')
    parser.add_argument('--slot', type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    config = AIConfig.from_env()
    if args.slot: config = replace(config, credentials=tuple(c for c in config.credentials if c.slot == args.slot))
    print(json.dumps({'configuration': config.public_status(), 'errors': config.errors, 'warnings': config.warnings}))
    if not args.catalogue and not args.live: return
    if not config.public_status()['available']:
        print(json.dumps({'live': 'skipped', 'reason': 'No usable credentials or invalid configuration.'}))
        return
    if args.catalogue:
        from google import genai
        from google.genai import types
        for credential in config.credentials:
            with genai.Client(api_key=credential.key, vertexai=False,
                              http_options=types.HttpOptions(timeout=8000, retry_options=types.HttpRetryOptions(attempts=1))) as client:
                for model in credential.models or config.models:
                    try:
                        info = client.models.get(model=model)
                        supported = 'generateContent' in (info.supported_actions or [])
                        print(json.dumps({'slot': credential.slot, 'model': model, 'generation_supported': supported}))
                    except Exception as exc:
                        print(json.dumps({'slot': credential.slot, 'model': model, 'failure': classify_error(exc).category}))
    if args.live:
        gateway = GeminiGateway(config)
        result = gateway.generate('Explain simply. Use plain text.', 'What is evaporation? Answer in two sentences.', 500)
        print(json.dumps({key: result.get(key) for key in ('status', 'model', 'credential_slot', 'attempts', 'latency_ms', 'error_category')}))
        gateway.close()
        if result['status'] != 'success': sys.exit(1)


if __name__ == '__main__': main()
