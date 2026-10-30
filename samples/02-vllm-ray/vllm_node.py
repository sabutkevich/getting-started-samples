"""Keep one Ray node alive for the complete vLLM serving lease."""
import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cluster_common import atomic_json, read_json, wait_json
from ray_node import alive, archive_logs, preflight, ray_command
from vllm_common import VLLM_VERSION, distributed_environment, inspect_model, stop_process_group, validate_topology


def serve_node(run, rank):
    config = read_json(run / "run.json")
    stop = run / "STOP"
    ray_process = service = None
    streams = []
    stage = "preflight"

    def update(value):
        atomic_json(run / f"status-{rank}.json", {"stage": value, "updated_at": time.time()})

    def interrupt(signum, frame):
        stop.touch()
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        import vllm
        import torch
        update(stage)
        if vllm.__version__ != VLLM_VERSION or not torch.version.cuda.startswith("13."):
            raise RuntimeError(f"Expected vLLM {VLLM_VERSION}/CUDA 13, got {vllm.__version__}/{torch.version.cuda}")
        manifest = inspect_model(config["model_dir"])
        validate_topology(config, manifest)
        if manifest != config["model_manifest"]:
            raise RuntimeError("Model files differ from the NFS checkpoint validated at submission")
        info = preflight(config, rank)
        from ep_preflight import check_ep_backend
        info["ep_preflight"] = check_ep_backend()
        if not any(p.get("link_layer") == "InfiniBand" and p.get("state", "").startswith("4:")
                   for p in info["rdma_ports"]):
            raise RuntimeError("No ACTIVE InfiniBand port available")
        info.update(vllm=vllm.__version__, model_manifest=manifest,
                    packages_sha256=hashlib.sha256((run / f"packages-{rank}.txt").read_bytes()).hexdigest())
        atomic_json(run / f"node-{rank}.json", info)
        print(json.dumps(info, indent=2), flush=True)
        peer = wait_json(run / f"node-{1-rank}.json", config["startup_timeout"], stop)
        if info["packages_sha256"] != peer["packages_sha256"]:
            raise RuntimeError("Installed dependencies differ between nodes; compare packages-0.txt and packages-1.txt")
        if info["ip"] == peer["ip"]:
            raise RuntimeError("Both jobs resolved to the same pod IP")
        if info["boot_hash"] and info["boot_hash"] == peer["boot_hash"]:
            raise RuntimeError("Same physical host boot ID; check include_nodes/platform placement")
        env = distributed_environment(config, [info, peer], rank, os.environ)
        env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET,GRAPH",
                   NCCL_DEBUG_FILE=str(run / "nccl.%h.%p.log"))
        hosts = [info["ip"], peer["ip"], "localhost", "127.0.0.1"]
        for name in ("NO_PROXY", "no_proxy"):
            env[name] = ",".join(filter(None, [env.get(name, ""), *hosts]))
        env.pop("RAY_ADDRESS", None)
        stage = "start-ray"
        update(stage)
        address = None if rank == 0 else wait_json(run / "head.json", config["startup_timeout"], stop)["address"]
        output = (run / f"ray-start-{rank}.log").open("w")
        streams.append(output)
        ray_process = subprocess.Popen(ray_command(config, rank, info["ip"], address),
                                       env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        if rank == 0:
            deadline = time.monotonic() + 120
            while not stop.exists():
                alive(ray_process, "Ray head")
                try:
                    with socket.create_connection((info["ip"], 6379), timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Ray head did not open port 6379")
                    time.sleep(1)
            if stop.exists():
                return
            head = {"address": f"{info['ip']}:6379", "ip": info["ip"], "dashboard": f"http://{info['ip']}:8265"}
            atomic_json(run / "head.json", head)
            output = (run / "service.log").open("w")
            streams.append(output)
            env["RAY_ADDRESS"] = head["address"]
            service = subprocess.Popen([sys.executable, "-u", str(run / "vllm_service.py"), "--run-dir", str(run)],
                                       env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        while not stop.exists():
            alive(ray_process, "Ray node")
            if service is not None and service.poll() is not None:
                raise RuntimeError(f"vLLM supervisor exited ({service.returncode}); see service.log / vllm.log")
            ready = read_json(run / "READY.json")
            stage = "serving" if ready else ("model-startup" if (run / "PASSED.json").exists() else "ray-gpu-test")
            update(stage)
            if ready and time.time() >= ready["expires_at"]:
                stop.write_text("Serving lease expired\n")
                break
            lifetime = config["startup_timeout"] + config["test_timeout"] + config["model_timeout"] + config["serve_seconds"] + 120
            if time.monotonic() - started > lifetime:
                raise TimeoutError("Node lifetime limit exceeded")
            time.sleep(3)
    except BaseException:
        if not stop.exists():
            atomic_json(run / f"node-error-{rank}.json", {"stage": stage, "rank": rank, "traceback": traceback.format_exc()})
            stop.touch()
        raise
    finally:
        if service:
            stop_process_group(service, timeout=30)
        if ray_process:
            try:
                subprocess.run(["ray", "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
            stop_process_group(ray_process)
        archive_logs(run, config, rank)
        for stream in streams:
            stream.close()
        atomic_json(run / f"stopped-{rank}.json", {"rank": rank, "stopped_at": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    serve_node(args.run_dir, args.rank)
