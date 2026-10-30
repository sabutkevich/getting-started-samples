"""Runs on Ray head. Verifies placement, CUDA, object transfer and NCCL via actors."""
import argparse
import json
import os
import socket
import time
import traceback
from pathlib import Path

from cluster_common import atomic_json, nccl_bus_bandwidth, read_json


def test_cluster(run):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    config = read_json(run / "run.json")
    head = read_json(run / "head.json")
    metadata = [read_json(run / f"node-{rank}.json") for rank in (0, 1)]
    expected_ips = {entry["ip"] for entry in metadata}
    timeout = config["test_timeout"]
    ray.init(address=head["address"], namespace="cloud-ray-" + config["id"])
    actors = []
    try:
        deadline = time.monotonic() + timeout
        while True:
            nodes = [node for node in ray.nodes() if node["Alive"]]
            if {node["NodeManagerAddress"] for node in nodes} == expected_ips:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Expected two Ray nodes at {expected_ips}; got {nodes}")
            time.sleep(2)
        for node in nodes:
            if node["Resources"].get("GPU", 0) != config["gpus_per_node"]:
                raise RuntimeError(f"Wrong GPU count registered by Ray: {node}")
        if len({entry["ray"] for entry in metadata}) != 1 or len({entry["torch"] for entry in metadata}) != 1:
            raise RuntimeError("Ray/PyTorch versions differ between nodes")

        @ray.remote(num_cpus=1, num_gpus=1)
        class GPUProbe:
            def inspect(self):
                import torch
                torch.cuda.set_device(0)
                value = (torch.ones((64, 64), device="cuda") @ torch.ones((64, 64), device="cuda"))
                assert bool(torch.all(value == 64).item()), "GPU calculation failed"
                torch.cuda.synchronize()
                prop = torch.cuda.get_device_properties(0)
                return {"node_id": ray.get_runtime_context().get_node_id(), "pod": socket.gethostname(),
                        "ray_gpu_ids": ray.get_gpu_ids(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                        "gpu": prop.name, "uuid": str(getattr(prop, "uuid", "unknown"))}

            def make_payload(self, length):
                import numpy as np
                return np.arange(length, dtype=np.uint32)

            def consume_payload(self, data):
                return {"length": len(data), "sum": int(data.sum(dtype="uint64")),
                        "node_id": ray.get_runtime_context().get_node_id()}

            def init_nccl(self, uri, rank, world_size):
                import torch
                import torch.distributed as dist
                from datetime import timedelta
                torch.cuda.set_device(0)
                dist.init_process_group("nccl", init_method=uri, rank=rank, world_size=world_size,
                                        timeout=timedelta(seconds=180), device_id=torch.device("cuda:0"))
                return {"rank": rank, "backend": dist.get_backend()}

            def allreduce(self, size, iterations):
                import torch
                import torch.distributed as dist
                world = dist.get_world_size()
                rank = dist.get_rank()
                x = torch.ones(size // 4, dtype=torch.float32, device="cuda")
                for _ in range(3):
                    dist.all_reduce(x)
                # Restore small values after warmup; powers of world stay exactly representable.
                x.fill_(rank + 1)
                dist.barrier(device_ids=[0])
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(iterations):
                    dist.all_reduce(x)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                expected = world * (world + 1) // 2 * world ** (iterations - 1)
                assert torch.allclose(x, torch.full_like(x, float(expected)), rtol=1e-5, atol=0), "NCCL AllReduce data check failed"
                return {"rank": rank, "elapsed_seconds": elapsed, "verified": True}

            def close(self):
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()

        # Pin GPU actors to each distinct Ray node, rather than relying on scheduler spreading.
        nodes.sort(key=lambda node: node["NodeManagerAddress"])
        for node in nodes:
            for _ in range(config["gpus_per_node"]):
                actor = GPUProbe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False)).remote()
                actors.append(actor)
        probes = ray.get([actor.inspect.remote() for actor in actors], timeout=timeout)
        uuids = [probe["uuid"] for probe in probes if probe["uuid"] != "unknown"]
        if len(uuids) != len(set(uuids)):
            raise RuntimeError("Ray actors were assigned the same physical GPU")
        print("GPU actors:", json.dumps(probes, indent=2), flush=True)

        length = 8 * 1024**2  # 32 MiB; checks Ray's object-manager path in BOTH directions.
        left, right = actors[0], actors[config["gpus_per_node"]]
        transfers = []
        for producer, consumer in ((left, right), (right, left)):
            ref = producer.make_payload.remote(length)
            received = ray.get(consumer.consume_payload.remote(ref), timeout=timeout)
            if received["length"] != length or received["sum"] != length * (length - 1) // 2:
                raise RuntimeError("Cross-node Ray object transfer failed")
            transfers.append(received)
        print("Ray cross-node object transfers: OK", flush=True)

        uri = (run / "nccl-store").as_uri()  # unique, previously nonexistent shared-NFS file
        ray.get([actor.init_nccl.remote(uri, rank, len(actors)) for rank, actor in enumerate(actors)], timeout=timeout)
        size = config["allreduce_mib"] * 1024**2
        iterations = config["allreduce_iterations"]
        timings = ray.get([actor.allreduce.remote(size, iterations) for actor in actors], timeout=timeout)
        elapsed = max(record["elapsed_seconds"] for record in timings)
        busbw = nccl_bus_bandwidth(size, iterations, elapsed, len(actors))
        report = {
            "status": "PASSED", "completed_at": time.time(),
            "checks": {"ray_nodes": 2, "gpu_actors": len(actors), "cuda_on_every_gpu": True,
                       "cross_node_ray_object_transfer_both_directions": True,
                       "nccl_allreduce_data_check": True},
            "preflight": metadata, "actors": probes,
            "allreduce": {"buffer_mib": config["allreduce_mib"], "iterations": iterations,
                          "busbw_GB_per_second": busbw, "busbw_Gbit_per_second": busbw * 8,
                          "timings": timings},
            "note": "This is a functional smoke test, not a maximum-bandwidth benchmark.",
        }
    finally:
        if actors:
            try:
                ray.get([actor.close.remote() for actor in actors], timeout=30)
            except Exception:
                pass
            for actor in actors:
                ray.kill(actor, no_restart=True)
        ray.shutdown()
    atomic_json(run / "PASSED.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        test_cluster(args.run_dir)
    except BaseException:
        atomic_json(args.run_dir / "smoke-error.json", {"traceback": traceback.format_exc()})
        raise
