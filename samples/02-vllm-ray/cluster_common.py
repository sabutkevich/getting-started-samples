"""Shared helpers. This module deliberately needs only the Python standard library."""
import json
import os
import socket
import time
from pathlib import Path

RAY_VERSION = "2.59.0"
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42"
SOURCE_FILES = ("cluster_common.py", "bootstrap_ray.py", "ray_node.py", "smoke_test.py", "requirements-ray.txt")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def wait_json(path, timeout, stop=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop and Path(stop).exists():
            raise RuntimeError("Cluster stop requested")
        value = read_json(path)
        if value is not None:
            return value
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {path}")


def pod_ipv4():
    for entry in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        address = entry[4][0]
        if not address.startswith("127.") and address != "0.0.0.0":
            return address
    raise RuntimeError("Cannot resolve the pod's routable IPv4 address")


def job_plan(config):
    nodes = config["nodes"]
    if len(nodes) != 2 or len(set(nodes)) != 2 or not all(nodes):
        raise ValueError("Exactly two DIFFERENT physical nodes are required")
    gpus = config["gpus_per_node"]
    if not isinstance(gpus, int) or not 1 <= gpus <= 8:
        raise ValueError("gpus_per_node must be an integer from 1 to 8")
    if config["cpus_per_node"] < gpus:
        raise ValueError("Need at least one Ray CPU per GPU actor")
    if config["placement"] not in ("pinned", "single-job"):
        raise ValueError("placement must be pinned or single-job")
    common = {
        "region": "SR009", "base_image": config["image"],
        "instance_type": f"h200.{gpus}gpu.{12*gpus}C.{244*gpus}G",
        "type": "binary", "processes_per_worker": 1, "internet": True,
    }
    if config["placement"] == "pinned":
        return [dict(common, n_workers=1, include_nodes=[node]) for node in nodes]
    return [dict(common, n_workers=2, include_nodes=nodes)]


def resolve_rank(value, environ=None):
    if value != "auto":
        rank = int(value)
    else:
        environ = os.environ if environ is None else environ
        raw = environ.get("OMPI_COMM_WORLD_RANK", environ.get("RANK"))
        if raw is None:
            raise RuntimeError("binary launcher did not supply OMPI_COMM_WORLD_RANK or RANK")
        rank = int(raw)
    if rank not in (0, 1):
        raise ValueError(f"Expected node rank 0 or 1, got {rank}")
    return rank


def nccl_bus_bandwidth(size_bytes, iterations, elapsed_seconds, world_size):
    if elapsed_seconds <= 0 or world_size < 2:
        raise ValueError("Invalid AllReduce measurement")
    algbw = size_bytes * iterations / elapsed_seconds / 1e9
    return algbw * 2 * (world_size - 1) / world_size
