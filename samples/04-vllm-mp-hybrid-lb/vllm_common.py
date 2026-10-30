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
VLLM_FILES = ("cluster_common.py", "vllm_common.py", "bootstrap_vllm.py", "vllm_node.py", "requirements-vllm.txt", "diagnose_vllm.py")


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
    from diagnose_vllm import failure_report
    sections.append(failure_report(run))
    for path in sorted(run.glob("*-error*.json")):
        sections.append(f"{path.name}:\n{tail_log(path)}")
    for name, lines in (("vllm.log", 200), ("vllm-worker.log", 100), ("service.log", 40), ("smoke.log", 30),
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
                              for key in ("o_groups", "num_attention_heads", "n_routed_experts",
                                          "engram_max_ngram_size", "engram_n_heads")},
            "model_type": config.get("model_type"), "quantization_config": config.get("quantization_config"),
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "index_sha256": hashlib.sha256(index_raw).hexdigest(),
            "shards": len(shards), "stored_bytes": total}


def validate_model_topology(manifest, config):
    if (config["tensor_parallel_size"], config["pipeline_parallel_size"],
        config["data_parallel_size"], config["data_parallel_size_local"],
        config["enable_expert_parallel"]) != (1, 1, 16, 8, True):
        raise ValueError("This launcher runs the official two-node DEP recipe: TP1 / PP1 / DP16 / EP16")


def server_command(config, rank, head_ip):
    # Recipe multi_node_dep + documented per-node DP ownership for hybrid LB.
    if rank not in (0, 1):
        raise ValueError("Expected node rank 0 or 1")
    command = [
        "vllm", "serve", config["model_dir"],
        "--tokenizer-mode", "deepseek_v41", "--enable-expert-parallel",
        "--data-parallel-hybrid-lb", "--data-parallel-size", "16",
        "--data-parallel-size-local", "8", "--data-parallel-address", head_ip,
        "--data-parallel-start-rank", str(rank * 8),
        "--data-parallel-rpc-port", str(config["dp_rpc_port"]),
        "--tool-call-parser", "deepseek_v41", "--enable-auto-tool-choice",
        "--reasoning-parser", "deepseek_v41", "--mm-encoder-tp-mode", "data",
        "--host", "0.0.0.0", "--port", str(config["api_port"]),
        "--served-model-name", config["served_model_name"],
    ]
    # No TP/PP override, eager mode, backend tuning, or model source patches.
    for key, flag in (("max_model_len", "--max-model-len"),
                      ("max_num_seqs", "--max-num-seqs"),
                      ("gpu_memory_utilization", "--gpu-memory-utilization")):
        if config.get(key) is not None:
            command.extend([flag, str(config[key])])
    return command


def native_environment(config, rank, ip, inherited):
    env = dict(inherited)
    # Avoid carrying tuning and DP/EP overrides from an older launcher.
    for name in list(env):
        if name.startswith(("RAY_", "VLLM_")) and name != "VLLM_CACHE_ROOT":
            env.pop(name)
    env.update(VLLM_ENGINE_READY_TIMEOUT_S=str(config["model_timeout"]),
               VLLM_HOST_IP=ip, VLLM_USE_RUST_FRONTEND="1",
               GLOO_SOCKET_IFNAME=config["socket_ifname"],
               NCCL_SOCKET_IFNAME=config["socket_ifname"], NCCL_SOCKET_FAMILY="AF_INET",
               NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET,GRAPH",
               NCCL_DEBUG_FILE=str(Path(config["run_dir"]) / "nccl.%h.%p.log"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
    env["VLLM_API_KEY"] = (Path(config["run_dir"]) / "api-key").read_text().strip()
    return env


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
