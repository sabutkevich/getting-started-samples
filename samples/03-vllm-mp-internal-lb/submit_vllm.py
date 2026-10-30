#!/usr/bin/env python3
"""Launch DeepSeek V4.1 Flash on 2 x 8 H200 using existing Cloud.ru containers."""
import argparse
import json
import re
import secrets
import shlex
import shutil
import time
import uuid
from pathlib import Path

from cluster_common import BASE_IMAGE, atomic_json, job_plan, read_json
from vllm_common import MODEL_PATH, VLLM_FILES, diagnostic_report, inspect_model, server_command, validate_model_topology


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=MODEL_PATH)
    parser.add_argument("--image", default=BASE_IMAGE)
    parser.add_argument("--nodes", nargs=2, default=["hpcsrv-201.sr009", "hpcsrv-202.sr009"])
    parser.add_argument("--nfs-root", type=Path, default=Path("/home/jovyan"))
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--socket-ifname", default="eth0", help="Pod TCP control interface; NCCL data uses InfiniBand")
    parser.add_argument("--dp-rpc-port", type=int, default=13345)
    parser.add_argument("--served-model-name", default="deepseek-v4.1-flash")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--startup-timeout", type=int, default=3600, help="Queue, install and node preflight; seconds")
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
    if not 1024 <= args.dp_rpc_port <= 65535 or args.dp_rpc_port == args.api_port:
        parser.error("--dp-rpc-port must be 1024..65535 and different from the HTTP port")
    if not 1024 <= args.api_port <= 65535:
        parser.error("--api-port must be 1024..65535")
    if any(value is not None and value <= 0 for value in (args.max_model_len, args.max_num_seqs, args.startup_timeout, args.model_timeout, args.serve_minutes)):
        parser.error("Lengths, sequence count and time limits must be positive")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be greater than zero and at most 1")
    return args


def prepare(args):
    config = {
        "application": "vllm", "id": uuid.uuid4().hex[:12], "image": args.image,
        "nodes": args.nodes, "gpus_per_node": 8, "cpus_per_node": 96, "placement": "pinned",
        "model_dir": str(Path(args.model_dir).expanduser().absolute()),
        "strategy": "multi_node_dep", "load_balancing": "internal", "tensor_parallel_size": 1,
        "data_parallel_size": 16, "data_parallel_size_local": 8, "pipeline_parallel_size": 1,
        "enable_expert_parallel": True, "runtime_patches": [], "dp_rpc_port": args.dp_rpc_port, "socket_ifname": args.socket_ifname,
        "startup_timeout": args.startup_timeout,
        "model_timeout": args.model_timeout, "serve_seconds": args.serve_minutes * 60,
        "api_port": args.api_port, "served_model_name": args.served_model_name,
        "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    plan = job_plan(config)
    if args.dry_run:
        print(json.dumps({"configuration": config, "job_requests": plan,
                          "server_commands": [server_command(config, rank, "<HEAD_POD_IP>") for rank in (0, 1)],
                          "model_validation": "Deferred: dry-run does not read weights"}, indent=2))
        return None, plan
    # Fail BEFORE requesting 16 GPUs if the checkpoint is missing or incomplete.
    config["model_manifest"] = inspect_model(config["model_dir"])
    config["model_dir"] = config["model_manifest"]["path"]
    validate_model_topology(config["model_manifest"], config)
    nfs = args.nfs_root.expanduser().resolve(strict=True)
    run = nfs / ("cloud-vllm-dep-single-api-" + config["id"])
    run.mkdir(mode=0o700)
    config["run_dir"] = str(run)
    for name in VLLM_FILES:
        shutil.copy2(Path(__file__).resolve().parent / name, run / name)
    atomic_json(run / "run.json", config)
    key_file = run / "api-key"
    key_file.write_text(secrets.token_urlsafe(32) + "\n")
    key_file.chmod(0o600)
    for rank, request in enumerate(plan):
        entry = run / f"launch-{rank}.sh"
        command = ["python", "-u", str(run / "bootstrap_vllm.py"), "--run-dir", str(run), "--rank", str(rank)]
        entry.write_text("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n")
        entry.chmod(0o755)
        request["script"] = str(entry)
    return run, plan


def status(run):
    report = {"run_dir": str(run)}
    for name in ("READY.json", "service-exit.json", "status-0.json", "status-1.json", "stopped-0.json", "stopped-1.json"):
        value = read_json(run / name)
        if value is not None:
            report[name] = value
    report["errors"] = [read_json(path) for path in sorted(run.glob("*-error*.json"))]
    print(json.dumps(report, indent=2))


def main(argv=None):
    args = arguments(argv)
    if args.stop:
        (args.stop / "STOP").write_text("vLLM stop requested\n")
        print("Shutdown requested. Both node supervisors will stop vLLM on both nodes.")
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
    print(f"Resources: 2 physical nodes x 8 H200 = 16 GPUs. Official multi_node_dep: TP1 / DP16 / EP16 / PP1; Rust frontend; stock vLLM 0.32.0; one API on head, internal LB over all 16 DP engines.", flush=True)
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return
    import client_lib

    jobs = []
    try:
        for request in plan:
            job = client_lib.Job(**request)
            jobs.append(job)
            job.submit()
        print("Waiting for install, native vLLM startup and first generated response...", flush=True)
        deadline = time.monotonic() + args.startup_timeout + args.model_timeout + 120
        next_update = 0
        while True:
            errors = list(run.glob("*-error*.json"))
            if errors:
                raise RuntimeError("\n".join(path.read_text() for path in errors))
            ready = read_json(run / "READY.json")
            if ready:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Startup deadline exceeded; check install-*.log / vllm.log / vllm-worker.log")
            if time.monotonic() >= next_update:
                stages = [read_json(run / f"status-{rank}.json") for rank in (0, 1)]
                print("Progress: " + json.dumps(stages), flush=True)
                next_update = time.monotonic() + 30
            time.sleep(3)
        print(json.dumps(ready, indent=2))
        print("The API stays running. Stop with:")
        print(shlex.join(["python3.12", str(Path(__file__).resolve()), "--stop", str(run)]))
    except BaseException:
        (run / "STOP").touch()
        for job in jobs:
            try:
                job.kill()
            except Exception as error:
                print(f"Cleanup: {error}", flush=True)
        for job in jobs:
            try:
                job.logs(tail=80)
            except Exception as error:
                print(f"Platform logs: {error}", flush=True)
        print("\nApplication diagnostics from NFS:\n" + diagnostic_report(run), flush=True)
        raise


if __name__ == "__main__":
    main()
