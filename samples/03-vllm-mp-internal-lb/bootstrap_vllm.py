"""Install a separate CUDA 13/vLLM environment inside each existing job container."""
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cluster_common import atomic_json, read_json, resolve_rank
from vllm_common import native_environment, stop_process_group
from cluster_common import pod_ipv4


def run_command(command, output, deadline, stop):
    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while process.poll() is None:
            if stop.exists():
                raise InterruptedError("Stop requested during installation")
            if time.monotonic() >= deadline:
                raise TimeoutError("Installation time limit exceeded")
            time.sleep(1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process.poll() is None:
            stop_process_group(process)


def cuda_environment(environment, inherited):
    """Find the CUDA 13 pip toolkit, rather than using CUDA 12 from the base image."""
    sites = list(environment.glob("lib/python*/site-packages"))
    nvccs = [p for site in sites for p in (site / "nvidia").glob("**/bin/nvcc")]
    if len(nvccs) != 1:
        raise RuntimeError(f"Expected one pip CUDA compiler, found {nvccs}")
    root = nvccs[0].parent.parent
    # CUDA 13 wheels share nvidia/cu13. Legacy layouts are deliberately rejected.
    if not (root / "include/cuda_runtime.h").is_file() or not (root / "nvvm/libdevice").is_dir():
        raise RuntimeError(f"Incomplete CUDA 13 toolkit at {root}; see install log")
    if not (root / "lib64").exists() and (root / "lib").is_dir():
        (root / "lib64").symlink_to("lib", target_is_directory=True)
    # Runtime wheels can contain SONAMEs only; nvcc/JIT link with -lcudart.
    for library in (root / "lib").glob("*.so.*"):
        link = library.with_name(library.name.split(".so.")[0] + ".so")
        if not link.exists():
            link.symlink_to(library.name)
    libraries = []
    for site in sites:
        for candidate in (site / "nvidia/cu13/lib", site / "torch/lib"):
            if candidate.is_dir():
                libraries.append(str(candidate))
        # cuDNN/NCCL can live outside the unified toolkit directory.
        libraries += [str(p) for p in (site / "nvidia").glob("*/lib") if p.is_dir() and str(p) not in libraries]
    # Preserve system RDMA/driver libraries; discard old CUDA/Conda runtime paths.
    inherited_libs = [p for p in inherited.get("LD_LIBRARY_PATH", "").split(":")
                      if p and not any(part in p.lower() for part in ("cuda", "conda", "site-packages", "stubs"))]
    # PyTorch's CUDA JIT cache does not create all missing parent directories.
    for cache in ("cache/torch/kernels", "cache/vllm", "cache/triton"):
        (environment / cache).mkdir(parents=True, exist_ok=True)
    env = dict(inherited, VIRTUAL_ENV=str(environment), CUDA_HOME=str(root), CUDA_PATH=str(root),
               CUDACXX=str(nvccs[0]), PATH=f"{environment}/bin:{root}/bin:" + inherited.get("PATH", ""),
               LD_LIBRARY_PATH=":".join(libraries + inherited_libs), PYTHONUNBUFFERED="1",
               XDG_CACHE_HOME=str(environment / "cache"),
               VLLM_CACHE_ROOT=str(environment / "cache/vllm"),
               TRITON_CACHE_DIR=str(environment / "cache/triton"))
    # Do not import packages from the base Conda environment through PYTHONPATH.
    for name in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "RAY_ADDRESS"):
        env.pop(name, None)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", required=True)
    args = parser.parse_args()
    run, rank = args.run_dir, resolve_rank(args.rank)
    stop = run / "STOP"
    try:
        config = read_json(run / "run.json")
        atomic_json(run / f"status-{rank}.json", {"stage": "install-vllm", "updated_at": time.time()})
        if not shutil.which("c++"):
            raise RuntimeError("Base image needs a C++ compiler for CUDA JIT")
        # Same absolute name in two separate container filesystems. vLLM may copy
        # CUDA_HOME/LD_LIBRARY_PATH from the driver to remote actors.
        environment = Path("/tmp") / ("cv-" + config["id"])
        environment.mkdir(mode=0o700)
        python = str(environment / "bin/python")
        deadline = time.monotonic() + config["startup_timeout"]
        with (run / f"install-{rank}.log").open("w") as output:
            # A clean venv is essential: base torch 2.9/CUDA 12.8 does not match this vLLM wheel.
            run_command([sys.executable, "-m", "venv", str(environment)], output, deadline, stop)
            run_command([python, "-m", "pip", "install", "--disable-pip-version-check", "uv==0.10.9"],
                        output, deadline, stop)
            run_command([str(environment / "bin/uv"), "pip", "install", "--python", python,
                         "--cache-dir", str(environment / "uv-cache"), "--torch-backend=cu130",
                         "-r", str(run / "requirements-vllm.txt")],
                        output, deadline, stop)
            run_command([python, "-m", "pip", "check"], output, deadline, stop)
            packages = subprocess.check_output([python, "-m", "pip", "freeze"], text=True, timeout=60)
            (run / f"packages-{rank}.txt").write_text(packages)
        env = cuda_environment(environment, os.environ)
        # Test actual compilation, not just nvcc's presence; this needs headers and libdevice.
        source = environment / "cuda-check.cu"
        source.write_text('#include <cuda_runtime.h>\n__global__ void test() {}\nint main() { return 0; }\n')
        with (run / f"install-{rank}.log").open("a") as output:
            subprocess.run([env["CUDACXX"], "--version"], env=env, stdout=output,
                           stderr=subprocess.STDOUT, check=True, timeout=30)
            subprocess.run([env["CUDACXX"], "--cudart=shared", "-arch=sm_90", str(source),
                            "-o", str(environment / "cuda-check")], env=env, stdout=output,
                           stderr=subprocess.STDOUT, check=True, timeout=120)
        atomic_json(run / f"environment-{rank}.json", {"venv": str(environment), "cuda_home": env["CUDA_HOME"],
                                                       "system_site_packages": False})
        env = native_environment(config, rank, pod_ipv4(), env)
        os.execve(python, [python, "-u", str(run / "vllm_node.py"), "--run-dir", str(run), "--rank", str(rank)], env)
    except BaseException:
        if not stop.exists():
            atomic_json(run / f"node-error-{rank}.json", {"stage": "install-vllm", "rank": rank,
                        "traceback": traceback.format_exc(), "install_log": str(run / f"install-{rank}.log")})
            stop.touch()
        atomic_json(run / f"stopped-{rank}.json", {"rank": rank, "stopped_at": time.time()})
        raise


if __name__ == "__main__":
    main()
