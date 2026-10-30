"""Official DEP with internal load balancing: API on head, headless remote engines."""
import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import traceback
from pathlib import Path

from cluster_common import atomic_json, pod_ipv4, read_json, wait_json
from vllm_common import (VLLM_VERSION, api_request, inspect_model, native_environment,
                         server_command, stop_process_group, validate_model_topology)


def preflight(config, rank):
    import torch
    import vllm
    if vllm.__version__ != VLLM_VERSION or not (torch.version.cuda or "").startswith("13."):
        raise RuntimeError(f"Expected vLLM {VLLM_VERSION} / CUDA 13; got {vllm.__version__} / {torch.version.cuda}")
    if torch.cuda.device_count() != 8:
        raise RuntimeError(f"Expected 8 visible CUDA GPUs; got {torch.cuda.device_count()}")
    manifest = inspect_model(config["model_dir"])
    validate_model_topology(manifest, config)
    if manifest != config["model_manifest"]:
        raise RuntimeError("Checkpoint manifest changed since submission")
    devices = []
    for index in range(8):
        with torch.cuda.device(index):
            x = torch.ones((64, 64), device=f"cuda:{index}")
            if not torch.all((x @ x) == 64).item():
                raise RuntimeError(f"CUDA matmul failed on GPU {index}")
            torch.cuda.synchronize()
            prop = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": prop.name, "uuid": str(getattr(prop, "uuid", "unknown"))})
    boot = Path("/proc/sys/kernel/random/boot_id")
    ports = []
    for port in sorted(Path("/sys/class/infiniband").glob("*/ports/*")):
        ports.append({"port": str(port), **{name: (port / name).read_text().strip()
                                          for name in ("state", "rate", "link_layer")}})
    if not any(p["link_layer"] == "InfiniBand" and p["state"].startswith("4:") for p in ports):
        raise RuntimeError("No ACTIVE InfiniBand port visible")
    return {"rank": rank, "requested_node": config["nodes"][rank], "pod": socket.gethostname(),
            "ip": pod_ipv4(), "vllm": vllm.__version__, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "nccl": list(torch.cuda.nccl.version()),
            "boot_hash": hashlib.sha256(boot.read_bytes()).hexdigest()[:16] if boot.exists() else None,
            "devices": devices, "rdma_ports": ports,
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True, timeout=30),
            "packages_sha256": hashlib.sha256((Path(config["run_dir"]) / f"packages-{rank}.txt").read_bytes()).hexdigest()}


def validate_peers(info, peer):
    if info["rank"] == peer["rank"] or info["ip"] == peer["ip"]:
        raise RuntimeError("Expected two distinct node ranks and pod IPs")
    if info["boot_hash"] and info["boot_hash"] == peer["boot_hash"]:
        raise RuntimeError("Both jobs are on the same physical host; inspect platform NODES INFO")
    if info["packages_sha256"] != peer["packages_sha256"]:
        raise RuntimeError("Installed dependencies differ between nodes")
    uuids = [d["uuid"] for entry in (info, peer) for d in entry["devices"] if d["uuid"] != "unknown"]
    if len(set(uuids)) != len(uuids):
        raise RuntimeError("GPU UUIDs overlap between processes/nodes")


def wait_api(process, base_url, key, model, deadline, stop):
    last_error = "API not started"
    while time.monotonic() < deadline:
        if stop.exists():
            raise InterruptedError("Stop requested during model startup")
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with status {process.returncode}; see vllm.log")
        try:
            api_request(base_url, "/health", key, timeout=3)
            models = api_request(base_url, "/v1/models", key, timeout=3)
            if model in [entry["id"] for entry in models.get("data", [])]:
                return
            last_error = f"Model not listed: {models}"
        except (OSError, RuntimeError) as error:
            last_error = str(error)
        time.sleep(3)
    raise TimeoutError(f"Model/API startup timed out: {last_error}")


def serve_node(run, rank):
    config = read_json(run / "run.json")
    stop = run / "STOP"
    process = None
    output = None
    stage = "preflight"
    reason = "error"

    def update(value):
        atomic_json(run / f"status-{rank}.json", {"stage": value, "updated_at": time.time()})

    def interrupt(signum, frame):
        stop.touch()
        raise InterruptedError(f"Node received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        update(stage)
        info = preflight(config, rank)
        atomic_json(run / f"node-{rank}.json", info)
        print(json.dumps(info, indent=2), flush=True)
        peer = wait_json(run / f"node-{1-rank}.json", config["startup_timeout"], stop)
        validate_peers(info, peer)
        if rank == 0:
            atomic_json(run / "head.json", {"ip": info["ip"]})
        head_ip = wait_json(run / "head.json", config["startup_timeout"], stop)["ip"]
        env = native_environment(config, rank, info["ip"], os.environ)
        for name in ("NO_PROXY", "no_proxy"):
            env[name] = ",".join(filter(None, [env.get(name, ""), head_ip, peer["ip"], info["ip"], "localhost", "127.0.0.1"]))
        command = server_command(config, rank, head_ip)
        atomic_json(run / f"server-command-{rank}.json", command)
        print("Starting:", json.dumps(command), flush=True)
        output = (run / ("vllm.log" if rank == 0 else "vllm-worker.log")).open("w")
        stage = "model-startup"
        update(stage)
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + config["model_timeout"]
        if rank == 0:
            base_url = f"http://{head_ip}:{config['api_port']}"
            wait_api(process, base_url, env["VLLM_API_KEY"], config["served_model_name"], deadline, stop)
            response = api_request(base_url, "/v1/chat/completions", env["VLLM_API_KEY"], {
                "model": config["served_model_name"], "messages": [{"role": "user", "content": "Reply briefly: what is 2 + 2?"}],
                "max_tokens": 128, "temperature": 0, "chat_template_kwargs": {"thinking": False},
            }, timeout=max(1, min(300, deadline - time.monotonic())))
            if not response.get("choices") or response.get("usage", {}).get("completion_tokens", 0) <= 0:
                raise RuntimeError(f"API did not generate tokens: {response}")
            ib = any("via NET/IB/" in path.read_text(errors="replace") for path in run.glob("nccl.*.log"))
            if not ib:
                raise RuntimeError("No vLLM NCCL NET/IB channel found; inspect nccl.*.log")
            ready = {"status": "READY", "completed_at": time.time(), "expires_at": time.time() + config["serve_seconds"],
                     "run_dir": str(run), "base_url": base_url, "openai_base_url": base_url + "/v1",
                     "endpoints": [base_url],
                     "model": config["served_model_name"], "api_key_file": str(run / "api-key"),
                     "physical_nodes": config["nodes"], "gpus": 16, "tensor_parallel_size": 1,
                     "pipeline_parallel_size": 1, "data_parallel_size": 16, "data_parallel_size_local": 8,
                     "expert_parallel_size": 16, "enable_expert_parallel": True,
                     "strategy": "multi_node_dep", "load_balancing": "internal", "runtime_patches": [],
                     "launcher": "native-multiprocessing", "frontend": "rust", "vllm_nccl_ib_channel_verified": ib,
                     "first_response": response}
            atomic_json(run / "READY.json", ready)
            print(json.dumps(ready, indent=2), flush=True)
        while not stop.exists():
            if process.poll() is not None:
                raise RuntimeError(f"vLLM node {rank} exited ({process.returncode}); inspect its log")
            ready = read_json(run / "READY.json")
            update("serving" if ready else "model-startup")
            if ready and time.time() >= ready["expires_at"]:
                reason = "lease-expired"
                stop.touch()
                break
            if not ready and time.monotonic() >= deadline:
                raise TimeoutError("Model startup exceeded its time limit")
            time.sleep(3)
        if reason != "lease-expired":
            reason = "stop-requested"
    except BaseException:
        if not stop.exists():
            atomic_json(run / f"node-error-{rank}.json", {"stage": stage, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        stop.touch()
        stop_process_group(process, timeout=30)
        if output:
            output.close()
        atomic_json(run / f"stopped-{rank}.json", {"rank": rank, "reason": reason, "stopped_at": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    serve_node(args.run_dir, args.rank)
