"""Offline deployment checks; no client_lib, Ray, PyTorch, CUDA or weights needed."""
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from bootstrap_vllm import cuda_environment, run_command
from cluster_common import atomic_json, read_json
from submit_vllm import arguments, prepare
from vllm_common import VLLM_FILES, diagnostic_report, distributed_environment, inspect_model, server_command, tail_log
from vllm_service import run_service, wait_api


def checkpoint(path):
    path.mkdir()
    atomic_json(path / "config.json", {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
                                      "text_config": {"o_groups": 8, "num_attention_heads": 64, "n_routed_experts": 384},
                                      "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"}})
    atomic_json(path / "model.safetensors.index.json", {"weight_map": {"layer.weight": "model-001.safetensors"}})
    (path / "model-001.safetensors").write_bytes(bytes(2048))
    (path / "tokenizer.json").write_text("{}")


class DeploymentTests(unittest.TestCase):
    def test_saved_application_error_visible_without_jobs_or_api_key(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "api-key").write_text("private-token")
            (run / "vllm.log").write_text("startup\nkey=private-token\nROOT CAUSE: kernel failed\n")
            atomic_json(run / "service-error.json", {"traceback": "vLLM exited with status 1"})
            report = diagnostic_report(run)
            self.assertIn("ROOT CAUSE: kernel failed", report)
            self.assertIn("vLLM exited with status 1", report)
            self.assertNotIn("private-token", report)
            self.assertIn("REDACTED", report)
            (run / "large.log").write_text("old output\n" * 10000 + "final error\n")
            self.assertEqual(tail_log(run / "large.log", 1, 64), "final error")

    def test_prepare_full_node_jobs_and_native_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="vllm test ") as temp:
            model = Path(temp) / "model with spaces"
            checkpoint(model)
            args = arguments(["--model-dir", str(model), "--nfs-root", temp, "--prepare-only"])
            run, jobs = prepare(args)
            config = read_json(run / "run.json")
            self.assertEqual([job["include_nodes"] for job in jobs], [["hpcsrv-201.sr009"], ["hpcsrv-202.sr009"]])
            self.assertTrue(all(job["instance_type"] == "h200.8gpu.96C.1952G" and job["n_workers"] == 1 for job in jobs))
            self.assertEqual(config["tensor_parallel_size"] * config["data_parallel_size"], 16)
            self.assertEqual(config["pipeline_parallel_size"], 1)
            self.assertEqual(config["expert_parallel_size"], 16)
            self.assertEqual(config["serve_seconds"], 7200)
            self.assertEqual(config["model_manifest"]["quantization_config"]["expert_dtype"], "fp4")
            for name in VLLM_FILES:
                self.assertTrue((run / name).is_file(), name)
            self.assertEqual((run.stat().st_mode & 0o777), 0o700)
            self.assertEqual(((run / "api-key").stat().st_mode & 0o777), 0o600)
            self.assertNotIn((run / "api-key").read_text().strip(), (run / "run.json").read_text())
            for rank, job in enumerate(jobs):
                command = shlex.split(Path(job["script"]).read_text().splitlines()[-1])
                self.assertEqual(command[3], str(run / "bootstrap_vllm.py"))
                self.assertEqual(command[-1], str(rank))
            command = server_command(config)
            self.assertEqual(command[2], str(model))
            self.assertIn("--enforce-eager", command)
            self.assertIn("--no-swa-bounded-replay", command)
            self.assertNotIn("--swa-bounded-replay", command)
            self.assertNotIn("--quantization", command)
            self.assertNotIn("--trust-remote-code", command)
            for flag, value in (("--data-parallel-size", "2"), ("--data-parallel-size-local", "1"),
                                ("--data-parallel-backend", "ray"), ("--pipeline-parallel-size", "1"),
                                ("--all2all-backend", "allgather_reducescatter"), ("--moe-backend", "humming")):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--enable-expert-parallel", command)

    def test_nested_download_directory_is_resolved_in_launch_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            outer = Path(temp) / "download"
            outer.mkdir()
            checkpoint(outer / "download")
            run, _ = prepare(arguments(["--model-dir", str(outer), "--nfs-root", temp, "--prepare-only"]))
            self.assertEqual(read_json(run / "run.json")["model_dir"], str(outer / "download"))

    def test_incompatible_checkpoint_geometry_rejected_before_gpu_request(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            checkpoint(model)
            data = read_json(model / "config.json")
            data["text_config"]["o_groups"] = 4
            atomic_json(model / "config.json", data)
            with self.assertRaisesRegex(ValueError, "o_groups=4"):
                prepare(arguments(["--model-dir", str(model), "--nfs-root", temp, "--prepare-only"]))
            self.assertFalse(list(Path(temp).glob("cloud-vllm-*")))

    def test_dp_placement_and_socket_environment_on_each_node(self):
        config = {"socket_ifname": "eth0", "model_timeout": 3600}
        metadata = [{"rank": 1, "ip": "10.1.0.2"}, {"rank": 0, "ip": "10.1.0.1"}]
        for rank in (0, 1):
            env = distributed_environment(config, metadata, rank, {"LD_LIBRARY_PATH": "/opt/rdma/lib", "VLLM_USE_RUST_FRONTEND": "1"})
            self.assertEqual(env["VLLM_HOST_IP"], f"10.1.0.{rank + 1}")
            self.assertEqual(env["VLLM_RAY_DP_PLACEMENT_NODE_IPS"], "10.1.0.1,10.1.0.2")
            self.assertEqual(env["VLLM_RAY_DP_PACK_STRATEGY"], "strict")
            self.assertEqual(env["VLLM_HUMMING_MOE_GEMM_TYPE"], "grouped")
            self.assertEqual(env["VLLM_USE_RUST_FRONTEND"], "0")
            self.assertEqual(env["NCCL_NET"], "IB")
            self.assertEqual(env["LD_LIBRARY_PATH"], "/opt/rdma/lib")

    def test_incomplete_checkpoint_rejected_before_creating_run(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            checkpoint(model)
            (model / "model-001.safetensors").write_text("version https://git-lfs.github.com/spec/v1")
            args = arguments(["--model-dir", str(model), "--nfs-root", temp, "--prepare-only"])
            with self.assertRaisesRegex(ValueError, "LFS pointer"):
                prepare(args)
            self.assertFalse(list(Path(temp).glob("cloud-vllm-*")))

    def test_wrong_architecture_and_escaping_shard_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            checkpoint(model)
            atomic_json(model / "model.safetensors.index.json", {"weight_map": {"w": "../other.safetensors"}})
            with self.assertRaisesRegex(ValueError, "Invalid shard path"):
                inspect_model(model)
            atomic_json(model / "config.json", {"architectures": ["DeepseekV3ForCausalLM"]})
            with self.assertRaisesRegex(ValueError, "DeepseekV41ForCausalLM"):
                inspect_model(model)

    def test_compiler_paths_preserve_rdma_but_remove_base_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            envdir = Path(temp)
            root = envdir / "lib/python3.12/site-packages/nvidia/cu13"
            for name in ("bin", "include", "lib", "nvvm/libdevice"):
                (root / name).mkdir(parents=True)
            (root / "bin/nvcc").touch()
            (root / "include/cuda_runtime.h").touch()
            env = cuda_environment(envdir, {"PATH": "/usr/bin", "PYTHONPATH": "/base/packages",
                                           "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/opt/rdma/lib:/home/user/conda/lib"})
            self.assertEqual(env["CUDA_HOME"], str(root))
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("/usr/local/cuda/lib64", env["LD_LIBRARY_PATH"])
            self.assertNotIn("/home/user/conda/lib", env["LD_LIBRARY_PATH"])
            self.assertIn("/opt/rdma/lib", env["LD_LIBRARY_PATH"])
            self.assertTrue((root / "lib64").is_dir())
            self.assertTrue((envdir / "cache/torch/kernels").is_dir())
            self.assertTrue((Path(env["VLLM_CACHE_ROOT"])).is_dir())

    def test_stop_interrupts_a_running_installer(self):
        with tempfile.TemporaryDirectory() as temp:
            stop = Path(temp) / "STOP"
            timer = threading.Timer(0.2, stop.touch)
            timer.start()
            try:
                with open(os.devnull, "w") as output:
                    with self.assertRaises(InterruptedError):
                        run_command([sys.executable, "-c", "import time; time.sleep(60)"], output,
                                    time.monotonic() + 10, stop)
            finally:
                timer.join()

    def test_api_readiness_rejects_early_process_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
            process.wait()
            with self.assertRaisesRegex(RuntimeError, "status 7"):
                wait_api(process, "http://127.0.0.1:1", "key", "model", time.monotonic() + 10, Path(temp) / "STOP")

    def test_full_service_readiness_generation_and_stop_without_gpu(self):
        # Real subprocess lifecycle; HTTP is stubbed because local sockets are sandboxed.
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            port = 8000
            atomic_json(run / "run.json", {"test_timeout": 10, "model_timeout": 20, "api_port": port,
                                           "served_model_name": "model", "serve_seconds": 30,
                                           "all2all_backend": "allgather_reducescatter",
                                           "nodes": ["node-a", "node-b"]})
            atomic_json(run / "head.json", {"ip": "127.0.0.1"})
            (run / "api-key").write_text("secret")
            (run / "nccl.fake.log").write_text("Channel 00: via NET/IB/0/GDRDMA\n")
            (run / "smoke_test.py").write_text("pass\n")
            fake = run / "fake.py"
            fake.write_text('''import os, time
from pathlib import Path
Path(__file__).with_suffix(".pid").write_text(str(os.getpid()))
time.sleep(60)
''')
            calls = []
            def request(base_url, path, key, payload=None, timeout=30):
                self.assertEqual(key, "secret")
                if not fake.with_suffix(".pid").exists():
                    raise OSError("API not started")
                calls.append(path)
                if path == "/v1/models":
                    return {"data": [{"id": "model"}]}
                if path == "/v1/chat/completions":
                    self.assertFalse((run / "READY.json").exists())
                    self.assertFalse(payload["chat_template_kwargs"]["thinking"])
                    return {"choices": [{"message": {"content": "4"}}], "usage": {"completion_tokens": 1}}
                return {}
            observations = []
            def stop_after_ready():
                until = time.monotonic() + 20
                while time.monotonic() < until:
                    ready = read_json(run / "READY.json")
                    if ready:
                        observations.append(ready)
                        (run / "STOP").touch()
                        return
                    time.sleep(0.1)
                (run / "STOP").touch()
            watcher = threading.Thread(target=stop_after_ready)
            previous = signal.getsignal(signal.SIGTERM)
            watcher.start()
            try:
                with patch("vllm_service.check_cluster"), patch("vllm_service.api_request", side_effect=request), patch("vllm_service.server_command", return_value=[sys.executable, str(fake)]):
                    run_service(run)
            finally:
                signal.signal(signal.SIGTERM, previous)
                watcher.join()
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0]["status"], "READY")
            self.assertEqual(observations[0]["pipeline_parallel_size"], 1)
            self.assertEqual(observations[0]["data_parallel_size"], 2)
            self.assertEqual(observations[0]["expert_parallel_size"], 16)
            self.assertEqual(observations[0]["first_response"]["choices"][0]["message"]["content"], "4")
            self.assertNotIn("secret", json.dumps(observations[0]))
            self.assertEqual(read_json(run / "service-exit.json")["reason"], "stop-requested")
            self.assertIn("/v1/chat/completions", calls)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(fake.with_suffix(".pid").read_text()), 0)


if __name__ == "__main__":
    unittest.main()
