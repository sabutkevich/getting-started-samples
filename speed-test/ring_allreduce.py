#!/usr/bin/env python3
"""Run NCCL Ring AllReduce on any number of SR009 H200 nodes.

Management Jupyter image: jupyter-cuda13.3-pt2.13.0-gpu:0.0.102.
Run this file with /usr/bin/python3.12; client_lib is supplied by Cloud.ru.
One separately pinned job is submitted per node, with one process per GPU.
GPU counts may differ between nodes; supported counts are 1..8 per H200 host.
The repository and run directory must be visible at identical NFS paths in jobs.

Examples:
    /usr/bin/python3.12 ring_allreduce.py
    /usr/bin/python3.12 ring_allreduce.py --nodes NODE0 NODE1 --gpus-per-node 8
    /usr/bin/python3.12 ring_allreduce.py --nodes NODE0 NODE1 NODE2 --gpus-per-node 8 4 2
    /usr/bin/python3.12 ring_allreduce.py --nodes NODE0 NODE1 --gpus-per-node 8 --dry-run

Reports are saved after every trial. Ring/IB with P2P and SHM disabled matches
the original network-only test; --allow-local-transports permits local P2P/SHM.
Reported bus bandwidth is a collective-equivalent metric, not a NIC counter.
Formula: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from collections import deque
from datetime import timedelta
from pathlib import Path


IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.9.0:0.0.42"


def save_json(path, value):
    """Readers see either the previous complete report or the new one."""
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def rank_layout(counts):
    """Assign contiguous global ranks to heterogeneous local GPU allocations."""
    layout = []
    for node_index, count in enumerate(counts):
        for local_rank in range(count):
            layout.append({"rank": len(layout), "node_index": node_index,
                           "local_rank": local_rank})
    return layout


def bandwidth(size_bytes, seconds, world_size):
    algorithm = size_bytes / seconds / 1e9
    bus = algorithm * 2 * (world_size - 1) / world_size
    return {"seconds_per_allreduce": seconds, "algbw_GB_s": algorithm,
            "busbw_GB_s": bus, "busbw_Gbit_s": bus * 8,
            "equivalent_rank_TX_plus_RX_Gbit_s": bus * 16}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", nargs="+", help="Distinct physical SR009 host names; prompt when omitted")
    parser.add_argument("--gpus-per-node", nargs="+", type=int,
                        help="One count for all nodes, or one count per node (1..8); default 1 with --nodes")
    parser.add_argument("--nfs-root", type=Path, default=Path("/home/jovyan"), help="Existing shared NFS root")
    parser.add_argument("--image", default=IMAGE, help="GPU job image; default: %(default)s")
    parser.add_argument("--sizes-mib", nargs="+", type=int, default=[64, 256, 1024], help="Buffer sizes per rank; default: %(default)s")
    parser.add_argument("--iterations", type=int, default=32, help="Timed AllReduce calls per trial; default: %(default)s")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup calls per buffer size; default: %(default)s")
    parser.add_argument("--trials", type=int, default=3, help="Trials per buffer size; default: %(default)s")
    parser.add_argument("--timeout-minutes", type=float, default=15, help="Total queue/startup/test deadline; default: %(default)s")
    parser.add_argument("--collective-timeout", type=int, default=600, help="Process-group timeout in seconds; default: %(default)s")
    parser.add_argument("--allow-local-transports", action="store_true", help="Allow P2P/NVLink and SHM; remote network remains IB")
    parser.add_argument("--socket-ifname", help="Optional NCCL TCP bootstrap interface, for example eth0")
    parser.add_argument("--dry-run", action="store_true", help="Print job/rank plan; create no files or jobs; requires --nodes")
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_node", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._node is not None or args._rank is not None:
        if args.run_dir is None or (args._node is not None and args._rank is not None):
            parser.error("Worker mode requires --run-dir and exactly one worker selector")
        return args
    if not args.nodes:
        if args.dry_run:
            parser.error("--dry-run requires --nodes")
        import client_lib
        print(client_lib.get_nodes(), flush=True)
        try:
            count = int(input("Number of physical nodes: "))
            if count <= 0:
                raise ValueError("Node count must be positive")
            args.nodes = [input(f"SR009 physical node {i}: ").strip() for i in range(count)]
            if args.gpus_per_node is None:
                args.gpus_per_node = [int(input(f"GPUs on {node} [1]: ") or "1") for node in args.nodes]
        except ValueError as error:
            parser.error(str(error))
    args.nodes = [node.strip() for node in args.nodes]
    if any(not node for node in args.nodes) or len(set(args.nodes)) != len(args.nodes):
        parser.error("Choose distinct, nonempty physical node names")
    counts = args.gpus_per_node or [1]
    if len(counts) == 1:
        counts = counts * len(args.nodes)
    if len(counts) != len(args.nodes) or any(not 1 <= value <= 8 for value in counts):
        parser.error("--gpus-per-node requires one value or one per node, each between 1 and 8")
    args.gpus_per_node = counts
    if min(args.iterations, args.trials, args.collective_timeout) <= 0 or args.warmup < 0:
        parser.error("Iterations, trials and collective timeout must be positive; warmup must be nonnegative")
    if not 0 < args.timeout_minutes < float("inf") or any(size <= 0 for size in args.sizes_mib):
        parser.error("Timeout must be positive and finite; buffer sizes must be positive")
    if not re.fullmatch(r"cr\.ai\.cloud\.ru/[^/]+/[^/:@]+(?::[^/\s]+|@sha256:[a-fA-F0-9]{64})", args.image):
        parser.error("--image must be a full cr.ai.cloud.ru image with a tag or digest")
    return args


def configuration(args):
    return {"nodes": args.nodes, "gpus_per_node": args.gpus_per_node,
            "world_size": sum(args.gpus_per_node), "layout": rank_layout(args.gpus_per_node),
            "image": args.image, "sizes_mib": args.sizes_mib,
            "iterations": args.iterations, "warmup": args.warmup, "trials": args.trials,
            "collective_timeout": args.collective_timeout, "timeout_seconds": args.timeout_minutes * 60,
            "allow_local_transports": args.allow_local_transports, "socket_ifname": args.socket_ifname}


def job_plan(config, run):
    requests = []
    for index, (node, gpus) in enumerate(zip(config["nodes"], config["gpus_per_node"])):
        disabled = "0" if config["allow_local_transports"] else "1"
        env = {"OMP_NUM_THREADS": "1", "NCCL_ALGO": "Ring", "NCCL_NET": "IB",
               "NCCL_IB_DISABLE": "0", "NCCL_P2P_DISABLE": disabled, "NCCL_SHM_DISABLE": disabled,
               "NCCL_SOCKET_FAMILY": "AF_INET", "NCCL_DEBUG": "INFO",
               "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH", "NCCL_DEBUG_FILE": str(run / "nccl.%h.%p.log")}
        if config["socket_ifname"]:
            env["NCCL_SOCKET_IFNAME"] = config["socket_ifname"]
        requests.append({"script": str(run / f"launch-{index}.sh"), "base_image": config["image"],
                         "region": "SR009", "instance_type": f"h200.{gpus}gpu.{12*gpus}C.{244*gpus}G",
                         "n_workers": 1, "processes_per_worker": 1, "type": "binary",
                         "include_nodes": [node], "env_variables": env})
    return requests


def prepare(config, root):
    root = root.expanduser().resolve(strict=True)
    run = root / ("ring_allreduce_" + uuid.uuid4().hex[:10])
    run.mkdir(mode=0o700)
    shutil.copy2(Path(__file__).resolve(), run / "worker.py")
    save_json(run / "run.json", config)
    for index in range(len(config["nodes"])):
        entry = run / f"launch-{index}.sh"
        command = ["python", "-u", str(run / "worker.py"), "--run-dir", str(run), "--_node", str(index)]
        entry.write_text("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n")
        entry.chmod(0o755)
    return run


def check_stop(run):
    if (run / "STOP").exists():
        raise RuntimeError("Stop requested")


def rank_worker(run, rank):
    import socket

    config = read_json(run / "run.json")
    placement = config["layout"][rank]
    local_rank = placement["local_rank"]
    world_size = config["world_size"]
    result = {"rank": rank, "placement": placement, "results": [], "status": "running"}
    progress = run / f"progress-{rank}.json"
    try:
        import torch
        import torch.distributed as dist

        check_stop(run)
        torch.set_num_threads(1)
        torch.cuda.set_device(local_rank)
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        info = dict(placement, requested_node=config["nodes"][placement["node_index"]],
                    pod=socket.gethostname(), gpu=torch.cuda.get_device_name(local_rank),
                    boot_hash=hashlib.sha256(boot.encode()).hexdigest())
        dist.init_process_group("nccl", init_method=(run / "store").as_uri(),
                                rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=config["collective_timeout"]))
        peers = [None] * world_size
        dist.all_gather_object(peers, info)
        physical = {}
        for peer in peers:
            key = peer["node_index"]
            if key in physical and physical[key] != peer["boot_hash"]:
                raise RuntimeError("Ranks of one job have different physical host identities")
            physical[key] = peer["boot_hash"]
        if len(set(physical.values())) != len(config["nodes"]):
            raise RuntimeError("Different requested nodes were placed on the same physical host")
        if rank == 0:
            save_json(run / "peers.json", peers)
        result["device"] = info
        timing = torch.empty(1, dtype=torch.float64, device="cuda")
        for mib in config["sizes_mib"]:
            check_stop(run)
            size = mib * 1024**2
            x = torch.empty(size // 4, dtype=torch.float32, device="cuda")
            measurements = []
            record = {"buffer_mib": mib, "trials": measurements}
            result["results"].append(record)
            # Timed reductions use zeros so repeated SUM cannot overflow FP32.
            x.zero_()
            for _ in range(config["warmup"]):
                dist.all_reduce(x)
            for trial in range(config["trials"]):
                check_stop(run)
                # Check nonzero rank-dependent data outside the timed interval.
                x.fill_(rank + 1)
                dist.all_reduce(x)
                expected = world_size * (world_size + 1) / 2
                if not torch.allclose(x, torch.full_like(x, expected), rtol=1e-5, atol=0):
                    raise RuntimeError(f"Nonzero data check failed on rank {rank}")
                x.zero_()
                dist.barrier(device_ids=[local_rank])
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(config["iterations"]):
                    dist.all_reduce(x, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                timing.fill_(time.perf_counter() - start)
                # Reduce timing only after measurement and use the slowest rank.
                dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                seconds = timing.item() / config["iterations"]
                if not bool(torch.all(x == 0).item()):
                    raise RuntimeError(f"Timed data check failed on rank {rank}")
                measurements.append(dict(trial=trial + 1, data_check=True,
                                         **bandwidth(size, seconds, world_size)))
                save_json(progress, result)
                if rank == 0:
                    print(f"{mib} MiB, trial {trial + 1}/{config['trials']}: "
                          f"busbw={measurements[-1]['busbw_GB_s']:.2f} GB/s", flush=True)
            record["median"] = bandwidth(size, statistics.median(
                entry["seconds_per_allreduce"] for entry in measurements), world_size)
            save_json(progress, result)
            del x
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()
        result.update(status="done", data_check=True)
        save_json(run / f"done-{rank}.json", result)
    except BaseException:
        save_json(run / f"error-{rank}.json", {"rank": rank, "traceback": traceback.format_exc()})
        (run / "STOP").touch()
        raise
    # The node supervisor terminates failed ranks; do not block in NCCL cleanup
    # after one peer has failed or disappeared.


def terminate_children(children):
    alive = [process for process in children if process.poll() is None]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in alive:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + (5 if sig == signal.SIGTERM else 2)
        for process in alive:
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        alive = [process for process in alive if process.poll() is None]


def node_worker(run, node_index):
    children, logs = [], []
    try:
        import torch

        config = read_json(run / "run.json")
        requested = config["gpus_per_node"][node_index]
        visible = torch.cuda.device_count()
        if visible != requested:
            raise RuntimeError(f"Node {node_index}: requested {requested} GPUs but sees {visible}")
        deadline = time.monotonic() + config["timeout_seconds"]
        for placement in config["layout"]:
            if placement["node_index"] != node_index:
                continue
            check_stop(run)
            rank = placement["rank"]
            output = (run / f"rank-{rank}.log").open("w")
            logs.append(output)
            children.append(subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--run-dir", str(run), "--_rank", str(rank)],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True))
        while True:
            codes = [process.poll() for process in children]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"Node {node_index} rank exited with an error; see rank-*.log")
            if all(code == 0 for code in codes):
                break
            check_stop(run)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Node {node_index} lifetime exceeded")
            time.sleep(1)
    except BaseException:
        save_json(run / f"error-node-{node_index}.json", {"traceback": traceback.format_exc()})
        (run / "STOP").touch()
        raise
    finally:
        terminate_children(children)
        for output in logs:
            output.close()


def report(config, run):
    ranks = [read_json(run / f"done-{rank}.json") for rank in range(config["world_size"])]
    save_json(run / "report.json", {"configuration": config, "ranks": ranks, "data_check": True})
    lines = [f"NCCL Ring AllReduce: {len(config['nodes'])} node(s), {config['world_size']} GPU(s)",
             "busbw = algbw * 2*(world_size-1)/world_size (collective equivalent, not NIC counters)."]
    if config["world_size"] == 1:
        lines.append("Single GPU: no inter-GPU communication; busbw is zero by definition.")
    for entry in ranks[0]["results"]:
        med = entry["median"]
        lines.append(f"{entry['buffer_mib']:4} MiB: algbw={med['algbw_GB_s']:.2f} GB/s; "
                     f"busbw={med['busbw_GB_s']:.2f} GB/s = {med['busbw_Gbit_s']:.2f} Gbit/s; "
                     f"equivalent per-rank TX+RX={med['equivalent_rank_TX_plus_RX_Gbit_s']:.2f} Gbit/s; "
                     f"AllReduce={med['seconds_per_allreduce'] * 1000:.2f} ms")
    lines.append("Data check: OK on every rank")
    text = "\n".join(lines) + "\n"
    (run / "report.txt").write_text(text)
    print(text, flush=True)


def launch(args):
    config = configuration(args)
    if args.dry_run:
        print(json.dumps({"configuration": config, "jobs": job_plan(config, args.nfs_root / "ring_allreduce_ID")}, indent=2))
        return
    import client_lib

    run = prepare(config, args.nfs_root)
    jobs = []
    print(f"Reports: {run}", flush=True)
    deadline = time.monotonic() + config["timeout_seconds"]
    try:
        for request in job_plan(config, run):
            check_stop(run)
            job = client_lib.Job(**request)
            jobs.append(job)
            job.submit()
            print(f"Submitted {request['include_nodes'][0]}: {request['instance_type']}", flush=True)
        while True:
            errors = sorted(run.glob("error-*.json"))
            if errors:
                raise RuntimeError("\n".join(f"{path.name}: {path.read_text()[:8000]}" for path in errors))
            finished = len(list(run.glob("done-*.json")))
            trials = sum(len(entry["trials"]) for path in run.glob("progress-*.json")
                         for entry in read_json(path)["results"])
            total = config["world_size"] * len(config["sizes_mib"]) * config["trials"]
            print(f"Progress: {trials}/{total} rank trials; {finished}/{config['world_size']} ranks finished; "
                  f"{max(0, deadline-time.monotonic()):.0f}s remaining", flush=True)
            if finished == config["world_size"]:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Overall deadline exceeded; check scheduling, rank logs and inter-job connectivity")
            time.sleep(5)
        report(config, run)
    except BaseException:
        (run / "STOP").touch()
        for job in jobs:
            try:
                job.kill()
            except Exception as error:
                print("Job cleanup:", error, flush=True)
        raise
    finally:
        for job in jobs:
            try:
                job.logs(tail=40)
            except Exception as error:
                print("Log retrieval:", error, flush=True)
        for path in sorted(run.glob("nccl.*.log")):
            with path.open(errors="replace") as stream:
                lines = deque((line.rstrip() for line in stream if any(
                    key in line for key in ("Using network", "GDRDMA", "speed=", "WARN"))), maxlen=20)
            print(f"\nNCCL: {path}\n" + "\n".join(lines), flush=True)


def main(argv=None):
    args = arguments(argv)
    if args._node is not None:
        node_worker(args.run_dir, args._node)
    elif args._rank is not None:
        rank_worker(args.run_dir, args._rank)
    else:
        launch(args)


if __name__ == "__main__":
    main()
