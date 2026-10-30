#!/usr/bin/env python3
"""Send a request from management Jupyter; no OpenAI/PyTorch dependencies."""
import argparse
import json
import time
from pathlib import Path

from cluster_common import read_json
from vllm_common import api_request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--node", type=int, choices=(0, 1), default=0, help="Hybrid LB API node")
    parser.add_argument("--prompt", default="Briefly explain tensor parallelism.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--thinking", action="store_true", help="Enable reasoning; increase --max-tokens for long traces")
    args = parser.parse_args(argv)
    ready = read_json(args.run_dir / "READY.json")
    if not ready:
        parser.error("READY.json is missing; wait for the first successful model response")
    if (args.run_dir / "STOP").exists() or time.time() >= ready["expires_at"]:
        parser.error("This service has stopped or its lease has expired")
    key = (args.run_dir / "api-key").read_text().strip()
    try:
        result = api_request(ready["endpoints"][args.node], "/v1/chat/completions", key, {
            "model": ready["model"], "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens, "temperature": 0.2,
            "chat_template_kwargs": {"thinking": args.thinking},
        }, timeout=args.timeout)
    except OSError as error:
        raise SystemExit(f"API network error: {error}. The Jupyter pod must reach the job pod IP shown in READY.json.") from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
