#!/usr/bin/env python3
"""Launch DeepSeek V4.1 Flash with configurable GPU topology on Cloud.ru."""
import argparse
import json
import re
import secrets
import shlex
import shutil
import time
import uuid
from pathlib import Path

from topology import add_parallel_options, resource_values, topology_from_args, submit_checked
from cluster_common import BASE_IMAGE, SOURCE_FILES, atomic_json, job_plan, read_json
from vllm_common import MODEL_PATH, VLLM_FILES, diagnostic_report, inspect_model, server_command, validate_topology


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_parallel_options(parser, default_tp=8)
    parser.add_argument("--model-dir", default=MODEL_PATH)
    parser.add_argument("--image", default=BASE_IMAGE)
    parser.add_argument("--nfs-root", type=Path, default=Path("/home/jovyan"))
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--socket-ifname", default="eth0", help="Pod TCP control interface; NCCL data uses InfiniBand")
    parser.add_argument("--served-model-name", default="deepseek-v4.1-flash")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--startup-timeout", type=int, default=3600, help="Queue, install and Ray startup; seconds")
    parser.add_argument("--model-timeout", type=int, default=3600, help="Weights/JIT/API startup; seconds")
    parser.add_argument("--serve-minutes", type=int, default=120, help="Maximum serving lease AFTER readiness")
    parser.add_argument("--dry-run", action="store_true", help="No model reads, files or jobs")
    parser.add_argument("--prepare-only", action="store_true", help="Validate model and write launch files; no jobs")
    parser.add_argument("--status", type=Path, metavar="RUN_DIR")
    parser.add_argument("--stop", type=Path, metavar="RUN_DIR")
    parser.add_argument("--diagnose", type=Path, metavar="RUN_DIR", help="Read saved NFS logs; no jobs or GPU requests")
    args = parser.parse_args(argv)
    if args.status or args.stop or args.diagnose:
        target = args.status or args.stop or args.diagnose
        if not (target / "run.json").is_file():
            parser.error("Run directory must contain run.json")
        return args
    if not re.fullmatch(r"cr\.ai\.cloud\.ru/[^/]+/[^/:@]+(?::[^/\s]+|@sha256:[a-fA-F0-9]{64})", args.image):
        parser.error("--image must be a cr.ai.cloud.ru image with a tag or digest")
    if not 1024 <= args.api_port <= 65535:
        parser.error("--api-port must be 1024..65535")
    if min(args.max_model_len, args.max_num_seqs, args.startup_timeout, args.model_timeout, args.serve_minutes) <= 0:
        parser.error("Lengths, sequence count and time limits must be positive")
    if not 0 < args.gpu_memory_utilization <= 0.95:
        parser.error("--gpu-memory-utilization must be greater than zero and at most 0.95")
    try:
        topology_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def prepare(args):
    config = {
        "application": "vllm", "id": uuid.uuid4().hex[:12], "image": args.image,
        **resource_values(args), **topology_from_args(args), "cpus_per_node": 12 * args.gpus_per_node,
        "model_dir": str(Path(args.model_dir).expanduser().absolute()),
        "all2all_backend": "allgather_reducescatter", "socket_ifname": args.socket_ifname,
        "startup_timeout": args.startup_timeout, "test_timeout": 600,
        "model_timeout": args.model_timeout, "serve_seconds": args.serve_minutes * 60,
        "keep_seconds": 0, "object_store_bytes": 2 * 1024**3,
        "allreduce_mib": 64, "allreduce_iterations": 8,
        "api_port": args.api_port, "served_model_name": args.served_model_name,
        "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs,
        "max_batched_tokens": min(4096, args.max_model_len),
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    plan = job_plan(config)
    if args.dry_run:
        print(json.dumps({"configuration": config, "job_requests": plan,
                          "server_command": server_command(config),
                          "model_validation": "Deferred: dry-run does not read weights"}, indent=2))
        return None, plan
    # Fail BEFORE requesting GPUs if the checkpoint is missing or incomplete.
    config["model_manifest"] = inspect_model(config["model_dir"])
    config["model_dir"] = config["model_manifest"]["path"]
    validate_topology(config, config["model_manifest"])
    nfs = args.nfs_root.expanduser().resolve(strict=True)
    run = nfs / ("cloud-vllm-ep-" + config["id"])
    run.mkdir(mode=0o700)
    config["run_dir"] = str(run)
    for name in SOURCE_FILES + VLLM_FILES:
        shutil.copy2(Path(__file__).resolve().parent / name, run / name)
    atomic_json(run / "run.json", config)
    key_file = run / "api-key"
    key_file.write_text(secrets.token_urlsafe(32) + "\n")
    key_file.chmod(0o600)
    for rank, request in enumerate(plan):
        entry = run / f"launch-{rank}.sh"
        command = ["python", "-u", str(run / "bootstrap_vllm.py"), "--run-dir", str(run), "--rank", "auto"]
        entry.write_text("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n")
        entry.chmod(0o755)
        request["script"] = str(entry)
    return run, plan


def status(run):
    report = {"run_dir": str(run)}
    for name in ("READY.json", "service-exit.json", "platform-job.json", *[path.name for pattern in ("status-*.json", "stopped-*.json") for path in sorted(run.glob(pattern))]):
        value = read_json(run / name)
        if value is not None:
            report[name] = value
    report["errors"] = [read_json(path) for path in sorted(run.glob("*-error*.json"))]
    print(json.dumps(report, indent=2))


def main(argv=None):
    args = arguments(argv)
    if args.stop:
        (args.stop / "STOP").write_text("vLLM stop requested\n")
        print("Shutdown requested. All configured node supervisors will stop vLLM/Ray.")
        return
    if args.status:
        status(args.status)
        return
    if args.diagnose:
        print(diagnostic_report(args.diagnose))
        return
    run, plan = prepare(args)
    if args.dry_run:
        return
    print(f"Run directory: {run}", flush=True)
    config = read_json(run / "run.json")
    print(f"Resources: {config['num_nodes']} nodes x {config['gpus_per_node']} H200; "
          f"TP={config['tensor_parallel_size']} DP={config['data_parallel_size']} "
          f"EP={config['expert_parallel_size']} PP={config['pipeline_parallel_size']}", flush=True)
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return
    import client_lib

    jobs = []
    try:
        for request in plan:
            job = client_lib.Job(**request)
            jobs.append(job)
            submit_checked(job, run)
        print("Waiting for install, Ray/NCCL, weights and first generated response...", flush=True)
        deadline = time.monotonic() + args.startup_timeout + 600 + args.model_timeout
        next_update = 0
        while True:
            errors = list(run.glob("*-error*.json"))
            if errors:
                raise RuntimeError("\n".join(path.read_text() for path in errors))
            ready = read_json(run / "READY.json")
            if ready:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Startup deadline exceeded; check install-*.log / smoke.log / vllm.log")
            if time.monotonic() >= next_update:
                stages = [read_json(run / f"status-{rank}.json") for rank in range(config["num_nodes"])]
                print("Progress: " + json.dumps(stages), flush=True)
                next_update = time.monotonic() + 30
            time.sleep(3)
        print(json.dumps(ready, indent=2))
        print("The API stays running. Stop with:")
        print(shlex.join(["python3.12", str(Path(__file__).resolve()), "--stop", str(run)]))
    except BaseException:
        (run / "STOP").touch()
        for job in jobs:
            if not getattr(job, "job_name", None):
                continue
            try:
                job.kill()
            except Exception as error:
                print(f"Cleanup: {error}", flush=True)
        for job in jobs:
            if not getattr(job, "job_name", None):
                continue
            try:
                job.logs(tail=80)
            except Exception as error:
                print(f"Platform logs: {error}", flush=True)
        print("\nApplication diagnostics from NFS:\n" + diagnostic_report(run), flush=True)
        raise


if __name__ == "__main__":
    main()
