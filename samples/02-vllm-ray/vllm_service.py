"""Head-only application: GPU smoke test, actual generation, bounded API service."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cluster_common import atomic_json, read_json
from vllm_common import api_request, server_command, stop_process_group


def check_cluster(run, config):
    """Release all probe GPU actors before vLLM requests its placement group."""
    import ray
    ray.init(address=read_json(run / "head.json")["address"])
    try:
        deadline = time.monotonic() + 120
        while ray.available_resources().get("GPU", 0) != 16:
            if (run / "STOP").exists():
                raise InterruptedError("Stop requested")
            if time.monotonic() >= deadline:
                raise TimeoutError("Probe actors did not release all 16 GPU resources")
            time.sleep(1)
    finally:
        ray.shutdown()


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
            if model not in [entry["id"] for entry in models.get("data", [])]:
                raise RuntimeError(f"Model {model} is not listed by the API: {models}")
            return
        except (OSError, RuntimeError) as error:
            last_error = str(error)
        time.sleep(3)
    raise TimeoutError(f"Model/API startup exceeded its limit: {last_error}")


def run_service(run):
    config = read_json(run / "run.json")
    stop = run / "STOP"
    process = smoke = None
    reason = "error"

    def interrupt(signum, frame):
        stop.touch()
        raise InterruptedError(f"Service received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        with (run / "smoke.log").open("w") as output:
            smoke = subprocess.Popen([sys.executable, "-u", str(run / "smoke_test.py"), "--run-dir", str(run)],
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + config["test_timeout"] + 60
            while smoke.poll() is None:
                if stop.exists():
                    raise InterruptedError("Stop requested during GPU test")
                if time.monotonic() > deadline:
                    raise TimeoutError("16-GPU Ray/NCCL smoke test exceeded time limit")
                time.sleep(2)
            if smoke.returncode:
                raise RuntimeError("16-GPU Ray/NCCL smoke test failed; see smoke.log")
        check_cluster(run, config)
        env = dict(os.environ, VLLM_API_KEY=(run / "api-key").read_text().strip(),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
        base_url = f"http://{read_json(run / 'head.json')['ip']}:{config['api_port']}"
        command = server_command(config)
        atomic_json(run / "server-command.json", command)
        print("Starting:", json.dumps(command), flush=True)
        with (run / "vllm.log").open("w") as output:
            process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + config["model_timeout"]
            wait_api(process, base_url, env["VLLM_API_KEY"], config["served_model_name"], deadline, stop)
            response = api_request(base_url, "/v1/chat/completions", env["VLLM_API_KEY"], {
                "model": config["served_model_name"], "messages": [{"role": "user", "content": "Reply briefly: what is 2 + 2?"}],
                "max_tokens": 128, "temperature": 0,
                "chat_template_kwargs": {"thinking": False},
            }, timeout=max(1, min(300, deadline - time.monotonic())))
            if not response.get("choices") or response.get("usage", {}).get("completion_tokens", 0) <= 0:
                raise RuntimeError(f"API responded without generated tokens: {response}")
            # Confirm the data plane actually used InfiniBand during the GPU smoke test.
            nccl_logs = list(run.glob("nccl.*.log"))
            ib = any("via NET/IB/" in path.read_text(errors="replace") for path in nccl_logs)
            if not ib:
                raise RuntimeError("No NCCL inter-node NET/IB channel found; inspect nccl.*.log")
            ready = {"status": "READY", "completed_at": time.time(),
                     "expires_at": time.time() + config["serve_seconds"], "run_dir": str(run),
                     "base_url": base_url, "openai_base_url": base_url + "/v1",
                     "api_key_file": str(run / "api-key"), "model": config["served_model_name"],
                     "physical_nodes": config["nodes"], "gpus": 16, "tensor_parallel_size": 8,
                     "pipeline_parallel_size": 1, "data_parallel_size": 2, "expert_parallel_size": 16,
                     "moe_backend": "humming", "all2all_backend": config["all2all_backend"],
                     "nccl_ib_channel_verified_in_gpu_smoke_test": ib,
                     "first_response": response}
            atomic_json(run / "READY.json", ready)
            print(json.dumps(ready, indent=2), flush=True)
            while not stop.exists() and time.time() < ready["expires_at"]:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited during serving ({process.returncode}); see vllm.log")
                time.sleep(2)
            reason = "stop-requested" if stop.exists() else "lease-expired"
    except BaseException:
        if not stop.exists():
            atomic_json(run / "service-error.json", {"traceback": traceback.format_exc()})
            stop.touch()
        else:
            reason = "stop-requested"
        if reason == "error":
            raise
    finally:
        stop_process_group(smoke)
        stop_process_group(process, timeout=30)
        atomic_json(run / "service-exit.json", {"reason": reason, "exited_at": time.time()})
        stop.touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_service(args.run_dir)
