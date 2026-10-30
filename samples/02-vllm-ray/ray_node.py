"""One supervisor per job node; owns Ray processes until test/lease completion."""
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

from cluster_common import RAY_VERSION, atomic_json, pod_ipv4, read_json, wait_json


def fingerprint(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def preflight(config, rank):
    import ray
    import torch

    if ray.__version__ != RAY_VERSION:
        raise RuntimeError(f"Ray {ray.__version__}, expected {RAY_VERSION}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: check driver/container compatibility and GPU allocation")
    count = torch.cuda.device_count()
    if count != config["gpus_per_node"]:
        raise RuntimeError(f"Expected {config['gpus_per_node']} visible GPU(s), got {count}")
    devices = []
    for index in range(count):
        with torch.cuda.device(index):
            x = torch.ones((64, 64), device=f"cuda:{index}")
            if not bool(torch.all(x @ x == 64).item()):
                raise RuntimeError("CUDA matmul check failed")
            torch.cuda.synchronize(index)
            properties = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": properties.name,
                            "uuid": str(getattr(properties, "uuid", "unknown"))})
    command = ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"]
    try:
        driver = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as error:
        driver = f"nvidia-smi unavailable: {error}"
    ports = []
    for port in sorted(Path("/sys/class/infiniband").glob("*/ports/*")):
        info = {"port": str(port)}
        for name in ("state", "rate", "link_layer"):
            try:
                info[name] = (port / name).read_text().strip()
            except OSError:
                pass
        ports.append(info)
    return {
        "rank": rank, "pod": socket.gethostname(), "ip": pod_ipv4(),
        "requested_node": config["nodes"][rank] if config["placement"] == "pinned" else None,
        "boot_hash": fingerprint("/proc/sys/kernel/random/boot_id"),
        "python": sys.version, "ray": ray.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
        "devices": devices, "driver": driver, "rdma_ports": ports,
    }


def ray_command(config, rank, ip, address=None):
    command = [
        "ray", "start", "--block", f"--node-ip-address={ip}",
        f"--num-cpus={config['cpus_per_node']}", f"--num-gpus={config['gpus_per_node']}",
        f"--object-store-memory={config['object_store_bytes']}",
        "--node-manager-port=62665", "--object-manager-port=62666",
        "--runtime-env-agent-port=62667", "--dashboard-agent-listen-port=62668",
        "--dashboard-agent-grpc-port=62669", "--metrics-export-port=62670",
        "--min-worker-port=21000", "--max-worker-port=21999",
    ]
    if rank == 0:
        command += ["--head", "--port=6379", "--include-dashboard=true",
                    "--dashboard-host=0.0.0.0", "--dashboard-port=8265",
                    "--ray-client-server-port=10001", f"--temp-dir=/tmp/rh-{config['id']}"]
    else:
        command.append(f"--address={address}")
    return command


def alive(process, description):
    if process.poll() is not None:
        raise RuntimeError(f"{description} exited with status {process.returncode}; see run logs")


def archive_logs(run, config, rank):
    root = Path(f"/tmp/rh-{config['id']}" if rank == 0 else "/tmp/ray") / "session_latest/logs"
    if not root.is_dir():
        return
    destination = run / f"ray-logs-{rank}"
    destination.mkdir(exist_ok=True)
    for path in root.iterdir():
        if path.is_file() and path.suffix in (".out", ".err", ".log"):
            try:
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 128 * 1024))
                    (destination / path.name).write_bytes(stream.read())
            except OSError:
                pass


def run_node(run, rank):
    config = read_json(run / "run.json")
    stop = run / "STOP"
    ray_process = test_process = None
    ray_log = test_log = None
    stage = "preflight"

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        info = preflight(config, rank)
        atomic_json(run / f"node-{rank}.json", info)
        print(json.dumps(info, indent=2), flush=True)
        peer = wait_json(run / f"node-{1-rank}.json", config["startup_timeout"], stop)
        if info["ip"] == peer["ip"]:
            raise RuntimeError("Both Ray nodes resolved to the same pod IP")
        if config["placement"] == "single-job":
            if info["boot_hash"] and info["boot_hash"] == peer["boot_hash"]:
                raise RuntimeError("Two workers may share a physical host. Use default --placement pinned")
        stage = "start-ray"
        address = None
        if rank == 1:
            address = wait_json(run / "head.json", config["startup_timeout"], stop)["address"]
        command = ray_command(config, rank, info["ip"], address)
        env = dict(os.environ, RAY_USAGE_STATS_ENABLED="0", PYTHONUNBUFFERED="1",
                   NCCL_NET="IB", NCCL_IB_DISABLE="0", NCCL_SOCKET_FAMILY="AF_INET",
                   NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET,GRAPH",
                   NCCL_DEBUG_FILE=str(run / "nccl.%h.%p.log"))
        env.pop("RAY_ADDRESS", None)
        ray_log = (run / f"ray-start-{rank}.log").open("w")
        ray_process = subprocess.Popen(command, stdout=ray_log, stderr=subprocess.STDOUT, env=env)
        if rank == 0:
            deadline = time.monotonic() + 120
            while True:
                alive(ray_process, "Ray head")
                if stop.exists():
                    return
                try:
                    with socket.create_connection((info["ip"], 6379), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Ray head GCS port did not open")
                    time.sleep(1)
            head = {"address": f"{info['ip']}:6379", "ip": info["ip"],
                    "dashboard": f"http://{info['ip']}:8265", "run_dir": str(run)}
            atomic_json(run / "head.json", head)
            stage = "smoke-test"
            test_log = (run / "smoke.log").open("w")
            test_process = subprocess.Popen([sys.executable, "-u", str(run / "smoke_test.py"),
                                             "--run-dir", str(run)], stdout=test_log,
                                            stderr=subprocess.STDOUT, env=env)
        started = time.monotonic()
        while not stop.exists():
            alive(ray_process, "Ray node supervisor")
            if test_process is not None and test_process.poll() is not None:
                if test_process.returncode != 0:
                    raise RuntimeError("Ray GPU smoke test failed; see smoke.log / smoke-error.json")
            result = read_json(run / "PASSED.json")
            if result:
                stage = "holding"
                if not config["keep_seconds"] or time.time() >= result["completed_at"] + config["keep_seconds"]:
                    break
            if time.monotonic() - started > config["startup_timeout"] + config["test_timeout"] + config["keep_seconds"]:
                raise TimeoutError("Node lifetime limit exceeded")
            time.sleep(2)
    except BaseException:
        atomic_json(run / f"node-error-{rank}.json", {
            "rank": rank, "stage": stage, "traceback": traceback.format_exc(),
        })
        stop.touch()
        raise
    finally:
        if test_process and test_process.poll() is None:
            test_process.terminate()
            try:
                test_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                test_process.kill()
                test_process.wait()
        if ray_process:
            # Jobs have their own containers; stop only this container's Ray processes.
            try:
                subprocess.run(["ray", "stop", "--force"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
            if ray_process.poll() is None:
                ray_process.terminate()
                try:
                    ray_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    ray_process.kill()
                    ray_process.wait()
        for stream in (ray_log, test_log):
            if stream:
                stream.close()
        archive_logs(run, config, rank)
        atomic_json(run / f"stopped-{rank}.json", {"rank": rank, "stopped_at": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    run_node(args.run_dir, args.rank)
