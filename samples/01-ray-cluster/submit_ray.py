#!/usr/bin/env python3
"""Submit Ray jobs from Cloud.ru's management Jupyter, without installing Ray there."""
import argparse
import json
import re
import shlex
import shutil
import time
import uuid
from pathlib import Path

from cluster_common import BASE_IMAGE, SOURCE_FILES, atomic_json, job_plan, read_json


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=BASE_IMAGE, help="Existing Cloud.ru base image; no custom build needed")
    parser.add_argument("--nodes", nargs=2, default=["hpcsrv-201.sr009", "hpcsrv-202.sr009"])
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--cpus-per-node", type=int, help="Defaults to 12 * GPUs per node")
    parser.add_argument("--nfs-root", type=Path, default=Path("/home/jovyan"))
    parser.add_argument("--placement", choices=("pinned", "single-job"), default="pinned")
    parser.add_argument("--startup-timeout", type=int, default=900, help="Seconds")
    parser.add_argument("--test-timeout", type=int, default=600, help="Seconds after Ray head is ready")
    parser.add_argument("--keep-alive", action="store_true", help="Keep the cluster after the test")
    parser.add_argument("--keep-minutes", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Print job requests without creating files or jobs")
    parser.add_argument("--prepare-only", action="store_true", help="Write NFS launch files without submitting jobs")
    parser.add_argument("--stop", type=Path, metavar="RUN_DIR", help="Request shutdown of an existing cluster")
    args = parser.parse_args(argv)
    if args.stop:
        if not (args.stop / "run.json").is_file():
            parser.error("--stop must point to an existing cloud-ray run directory")
        return args
    if not re.fullmatch(
        r"cr\.ai\.cloud\.ru/[^/]+/[^/:@]+(?::[^/\s]+|@sha256:[a-fA-F0-9]{64})", args.image
    ):
        parser.error("--image must be a full cr.ai.cloud.ru image with a tag or digest")
    if min(args.startup_timeout, args.test_timeout, args.keep_minutes) <= 0:
        parser.error("Timeouts and keep-minutes must be positive")
    return args


def prepare(args):
    config = {
        "id": uuid.uuid4().hex[:12], "image": args.image, "nodes": args.nodes,
        "gpus_per_node": args.gpus_per_node,
        "cpus_per_node": args.cpus_per_node if args.cpus_per_node is not None else 12 * args.gpus_per_node,
        "placement": args.placement,
        "startup_timeout": args.startup_timeout, "test_timeout": args.test_timeout,
        "keep_seconds": args.keep_minutes * 60 if args.keep_alive else 0,
        "object_store_bytes": 1024**3, "allreduce_mib": 256, "allreduce_iterations": 16,
    }
    plan = job_plan(config)
    if args.dry_run:
        print(json.dumps({"configuration": config, "job_requests": plan}, indent=2))
        return None, plan
    nfs = args.nfs_root.expanduser().resolve(strict=True)
    run = nfs / ("cloud-ray-" + config["id"])
    run.mkdir(mode=0o700)
    source = Path(__file__).resolve().parent
    for name in SOURCE_FILES:
        shutil.copy2(source / name, run / name)
    config["run_dir"] = str(run)
    atomic_json(run / "run.json", config)
    for index, request in enumerate(plan):
        rank = str(index) if args.placement == "pinned" else "auto"
        command = shlex.join(["python", "-u", str(run / "bootstrap_ray.py"),
                              "--run-dir", str(run), "--rank", rank])
        entry = run / f"launch-{index}.sh"
        entry.write_text("#!/bin/bash\nset -euo pipefail\nexec " + command + "\n")
        entry.chmod(0o755)
        request["script"] = str(entry)
    return run, plan


def request_stop(run):
    (run / "STOP").write_text("Stop requested from the management Jupyter\n")


def main(argv=None):
    args = arguments(argv)
    if args.stop:
        request_stop(args.stop)
        print(f"Shutdown requested. Node supervisors will stop Ray: {args.stop}")
        return
    run, plan = prepare(args)
    if args.dry_run:
        return
    print(f"Run directory: {run}", flush=True)
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return
    # Cloud.ru supplies this module in the management Jupyter.
    import client_lib

    jobs = []
    try:
        for request in plan:
            job = client_lib.Job(**request)
            jobs.append(job)
            job.submit()
        print("Waiting for Ray registration, GPU actors, object transfer and NCCL...", flush=True)
        deadline = time.monotonic() + args.startup_timeout + args.test_timeout
        while True:
            errors = list(run.glob("*-error*.json"))
            if errors:
                raise RuntimeError("\n".join(p.read_text() for p in errors))
            result = read_json(run / "PASSED.json")
            if result:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"No result before deadline; see logs in {run}")
            time.sleep(3)
        print(json.dumps(result, indent=2))
        if args.keep_alive:
            cluster = read_json(run / "head.json")
            print(f"Ray remains active for at most {args.keep_minutes} minutes after the test.")
            print(f"Ray address: {cluster['address']}; private dashboard: {cluster['dashboard']}")
            print("Stop: " + shlex.join(["python", str(Path(__file__).resolve()), "--stop", str(run)]))
        else:
            request_stop(run)
            stop_deadline = time.monotonic() + 90
            while not all((run / f"stopped-{r}.json").exists() for r in (0, 1)):
                if time.monotonic() >= stop_deadline:
                    raise TimeoutError("Ray test passed but node cleanup did not finish")
                time.sleep(2)
            print("Ray stopped on both nodes; binary jobs are finishing.")
    except BaseException:
        request_stop(run)
        for job in jobs:
            try:
                job.kill()
            except Exception as error:
                print(f"Job cleanup: {error}", flush=True)
        raise
    finally:
        for job in jobs:
            try:
                job.logs(tail=60)
            except Exception as error:
                print(f"Platform logs: {error}", flush=True)


if __name__ == "__main__":
    main()
