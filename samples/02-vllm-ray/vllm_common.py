"""Model, command and HTTP helpers; safe to import in the management Jupyter."""
import hashlib
import json
import os
import signal
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

VLLM_VERSION = "0.32.0"
MODEL_PATH = "/home/jovyan/MODELS/deepseek-ai_DeepSeek-V4.1-Flash-1017/deepseek-ai_DeepSeek-V4.1-Flash-1017"
VLLM_FILES = ("vllm_common.py", "bootstrap_vllm.py", "vllm_node.py", "vllm_service.py", "ep_preflight.py", "requirements-vllm.txt")


def tail_log(path, lines=200, max_bytes=64 * 1024):
    """Read bounded output even when model/worker logs are large."""
    path = Path(path)
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - max_bytes))
            return "\n".join(stream.read(max_bytes).decode(errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return "(file not created)"


def diagnostic_report(run):
    """Useful logs live on NFS, rather than in the platform supervisor output."""
    run = Path(run)
    sections = []
    for path in sorted(run.glob("*-error*.json")):
        sections.append(f"{path.name}:\n{tail_log(path)}")
    for name, lines in (("vllm.log", 200), ("service.log", 40), ("smoke.log", 30),
                        ("install-0.log", 30), ("install-1.log", 30)):
        if (run / name).exists():
            sections.append(f"{name} (last {lines} lines):\n{tail_log(run / name, lines)}")
    report = "\n\n".join(sections) or "No error files or logs created yet."
    try:
        key = (run / "api-key").read_text().strip()
    except FileNotFoundError:
        key = ""
    return report.replace(key, "[API KEY REDACTED]") if key else report


def inspect_model(directory):
    """Validate local files without loading weights or requiring transformers."""
    directory = Path(directory).expanduser().resolve(strict=True)
    # The downloaded checkpoint in this workspace has a repeated directory name.
    if not (directory / "config.json").exists() and (directory / directory.name / "config.json").is_file():
        directory = directory / directory.name
    raw = (directory / "config.json").read_bytes()
    config = json.loads(raw)
    architectures = config.get("architectures", [])
    if "DeepseekV41ForCausalLM" not in architectures:
        raise ValueError(f"Expected DeepseekV41ForCausalLM; got {architectures}")
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"Sharded checkpoint index missing: {index_path}")
    index_raw = index_path.read_bytes()
    index = json.loads(index_raw)
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise ValueError("Checkpoint index has no weights")
    total = 0
    for name in shards:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid shard path: {name}")
        shard = directory / relative
        if not shard.is_file() or shard.stat().st_size < 1024:
            raise ValueError(f"Shard missing, incomplete or an LFS pointer: {shard}")
        total += shard.stat().st_size
    if not (directory / "tokenizer.json").is_file():
        raise ValueError("tokenizer.json is missing")
    return {"path": str(directory), "architectures": architectures,
            "text_geometry": {key: config.get("text_config", config).get(key)
                              for key in ("o_groups", "num_attention_heads", "n_routed_experts")},
            "model_type": config.get("model_type"), "quantization_config": config.get("quantization_config"),
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "index_sha256": hashlib.sha256(index_raw).hexdigest(),
            "shards": len(shards), "stored_bytes": total}


def validate_topology(config, manifest):
    tp, dp, pp = (config[key] for key in ("tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size"))
    if (tp, dp, pp) != (8, 2, 1) or not config["enable_expert_parallel"]:
        raise ValueError("This launcher requires TP=8, DP=2, EP=16, PP=1")
    for key, divisor in (("o_groups", tp), ("num_attention_heads", tp), ("n_routed_experts", tp * dp)):
        value = manifest["text_geometry"].get(key)
        if not isinstance(value, int) or value < divisor or value % divisor:
            raise ValueError(f"Checkpoint {key}={value} is incompatible with divisor {divisor}")


def distributed_environment(config, metadata, rank, inherited):
    by_rank = {entry["rank"]: entry for entry in metadata}
    return dict(inherited, NCCL_NET="IB", NCCL_IB_DISABLE="0", NCCL_SOCKET_FAMILY="AF_INET",
                GLOO_SOCKET_IFNAME=config["socket_ifname"], NCCL_SOCKET_IFNAME=config["socket_ifname"],
                VLLM_HOST_IP=by_rank[rank]["ip"],
                VLLM_ENGINE_READY_TIMEOUT_S=str(config["model_timeout"]),
                VLLM_RAY_DP_PACK_STRATEGY="strict",
                VLLM_RAY_DP_PLACEMENT_NODE_IPS=",".join(by_rank[i]["ip"] for i in (0, 1)),
                VLLM_HUMMING_MOE_GEMM_TYPE="grouped", VLLM_USE_RUST_FRONTEND="0",
                RAY_DEDUP_LOGS="0")


def server_command(config):
    return [
        "vllm", "serve", config["model_dir"],
        "--host", "0.0.0.0", "--port", str(config["api_port"]),
        "--served-model-name", config["served_model_name"],
        "--distributed-executor-backend", "ray",
        "--tensor-parallel-size", str(config["tensor_parallel_size"]),
        "--pipeline-parallel-size", str(config["pipeline_parallel_size"]),
        "--data-parallel-size", str(config["data_parallel_size"]),
        "--data-parallel-size-local", "1", "--data-parallel-backend", "ray",
        "--enable-expert-parallel", "--all2all-backend", config["all2all_backend"],
        "--moe-backend", "humming", "--api-server-count", "1",
        "--engram-config", '{"cpu_offload":true,"embedding_across_dp":false}',
        "--tokenizer-mode", "deepseek_v41",
        "--tool-call-parser", "deepseek_v41", "--enable-auto-tool-choice",
        "--reasoning-parser", "deepseek_v41",
        "--language-model-only", "--enforce-eager",
        # Keep the first functional run independent of bounded-replay scheduling.
        "--no-swa-bounded-replay",
        "--max-model-len", str(config["max_model_len"]),
        "--max-num-seqs", str(config["max_num_seqs"]),
        "--max-num-batched-tokens", str(config["max_batched_tokens"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
    ]


def api_request(base_url, path, key, payload=None, timeout=30):
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Authorization": "Bearer " + key}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base_url.rstrip("/") + path, data=body, headers=headers)
    # Never route pod-local requests through an inherited HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read()
            return json.loads(data) if data else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:4000]
        raise RuntimeError(f"HTTP {error.code} from {path}: {detail}") from error


def stop_process_group(process, timeout=20):
    if process is None:
        return
    # Includes API subprocesses spawned by vLLM even if the group leader has exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
