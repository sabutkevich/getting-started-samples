"""Executed INSIDE each binary job: install Ray locally, keep base CUDA/PyTorch."""
import argparse
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from cluster_common import atomic_json, read_json, resolve_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", required=True)
    args = parser.parse_args()
    rank = resolve_rank(args.rank)
    run = args.run_dir
    try:
        config = read_json(run / "run.json")
        environment = Path(tempfile.mkdtemp(prefix=f"cry-{config['id'][:6]}-{rank}-", dir="/tmp"))
        log = run / f"install-{rank}.log"
        print(f"Rank {rank}: installing Ray in {environment}; log: {log}", flush=True)
        with log.open("w") as output:
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
                           check=True, stdout=output, stderr=subprocess.STDOUT, timeout=120)
            # Prevent the resolver from replacing the working platform PyTorch.
            torch_pin = subprocess.check_output(
                [sys.executable, "-c", "import torch; print('torch==' + torch.__version__)"], text=True
            )
            constraint = environment / "constraints.txt"
            constraint.write_text(torch_pin)
            python = environment / "bin/python"
            subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                            "--no-cache-dir", "-c", str(constraint), "-r", str(run / "requirements-ray.txt")],
                           check=True, stdout=output, stderr=subprocess.STDOUT,
                           timeout=min(config["startup_timeout"], 600))
            packages = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
            (run / f"packages-{rank}.txt").write_text(packages)
        env = dict(os.environ, PATH=str(environment / "bin") + ":" + os.environ.get("PATH", ""),
                   VIRTUAL_ENV=str(environment), PYTHONUNBUFFERED="1", RAY_USAGE_STATS_ENABLED="0")
        os.execve(str(python), [str(python), "-u", str(run / "ray_node.py"),
                               "--run-dir", str(run), "--rank", str(rank)], env)
    except BaseException:
        atomic_json(run / f"node-error-{rank}.json", {
            "stage": "install-ray", "rank": rank, "traceback": traceback.format_exc(),
            "install_log": str(run / f"install-{rank}.log"),
        })
        raise


if __name__ == "__main__":
    main()
