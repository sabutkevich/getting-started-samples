"""Offline tests for Cloud.ru placement and the official DEP launcher."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cluster_common import atomic_json, read_json
from submit_vllm import arguments, prepare
from vllm_common import diagnostic_report, native_environment, server_command
from vllm_node import serve_node, validate_peers, wait_api


def checkpoint(path):
    path.mkdir(parents=True)
    atomic_json(path / "config.json", {"architectures": ["DeepseekV41ForCausalLM"],
                                      "text_config": {"o_groups": 8, "engram_max_ngram_size": 4, "engram_n_heads": 8}, "model_type": "deepseek_v41"})
    atomic_json(path / "model.safetensors.index.json", {"weight_map": {"w": "model-001.safetensors"}})
    (path / "model-001.safetensors").write_bytes(bytes(2048))
    (path / "tokenizer.json").write_text("{}")


def metadata(rank):
    return {"rank": rank, "ip": f"10.1.0.{rank+1}", "boot_hash": f"boot-{rank}",
            "packages_sha256": "same", "devices": [{"uuid": f"gpu-{rank}-{i}"} for i in range(8)]}


class NativeTests(unittest.TestCase):
    def test_official_dep_assigns_disjoint_ranks_and_preserves_stock_runtime(self):
        with tempfile.TemporaryDirectory(prefix="dep test ") as temp:
            outer = Path(temp) / "model"
            checkpoint(outer / "model")
            run, jobs = prepare(arguments(["--model-dir", str(outer), "--nfs-root", temp, "--prepare-only"]))
            config = read_json(run / "run.json")
            self.assertEqual(config["model_dir"], str(outer / "model"))
            self.assertEqual(config["runtime_patches"], [])
            self.assertEqual(config["tensor_parallel_size"], 1)
            self.assertEqual(config["pipeline_parallel_size"], 1)
            self.assertEqual(config["data_parallel_size"], 16)
            self.assertEqual([r["include_nodes"] for r in jobs], [["hpcsrv-201.sr009"], ["hpcsrv-202.sr009"]])
            self.assertTrue(all(r["n_workers"] == 1 and r["instance_type"] == "h200.8gpu.96C.1952G" for r in jobs))
            self.assertEqual((run / "api-key").stat().st_mode & 0o777, 0o600)
            self.assertFalse((run / "patch_vllm_pp.py").exists())
            self.assertNotIn("patch_vllm_pp", (run / "bootstrap_vllm.py").read_text())
            ownership = []
            for rank in (0, 1):
                command = server_command(config, rank, "10.1.0.1")
                for flag in ("--enable-expert-parallel",):
                    self.assertIn(flag, command)
                for flag, value in (("--data-parallel-size", "16"), ("--data-parallel-size-local", "8"),
                                    ("--data-parallel-address", "10.1.0.1"), ("--data-parallel-rpc-port", "13345"),
                                    ("--mm-encoder-tp-mode", "data")):
                    self.assertEqual(command[command.index(flag)+1], value)
                if rank == 0:
                    self.assertNotIn("--headless", command)
                    self.assertNotIn("--data-parallel-start-rank", command)
                else:
                    self.assertIn("--headless", command)
                    self.assertEqual(command[command.index("--data-parallel-start-rank")+1], "8")
                ownership.append(set(range(rank * 8, rank * 8 + 8)))
                for flag in ("--data-parallel-hybrid-lb", "--data-parallel-external-lb", "--data-parallel-rank", "--nnodes", "--node-rank", "--tensor-parallel-size",
                             "--pipeline-parallel-size", "--no-swa-bounded-replay", "--enforce-eager",
                             "--distributed-executor-backend", "--engram-config", "--moe-backend",
                             "--max-model-len", "--max-num-seqs", "--gpu-memory-utilization"):
                    self.assertNotIn(flag, command)
            self.assertFalse(ownership[0] & ownership[1])
            self.assertEqual(ownership[0] | ownership[1], set(range(16)))

    def test_incomplete_weights_rejected_before_run_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            checkpoint(model)
            (model / "model-001.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "Shard missing"):
                prepare(arguments(["--model-dir", str(model), "--nfs-root", temp]))
            self.assertFalse(list(Path(temp).glob("cloud-vllm-dep-single-api-*")))

    def test_overrides_are_explicit_and_symmetric(self):
        config = {"model_dir": "/model", "api_port": 8000, "served_model_name": "model",
                  "max_model_len": 16384, "max_num_seqs": 8, "gpu_memory_utilization": 0.85,
                  "dp_rpc_port": 13345}
        for rank in (0, 1):
            command = server_command(config, rank, "10.1.0.1")
            self.assertEqual(command[command.index("--max-model-len")+1], "16384")
            self.assertEqual(command[command.index("--gpu-memory-utilization")+1], "0.85")

    def test_stale_ray_ep_environment_removed_and_rust_enabled(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "api-key").write_text("secret")
            config = {"run_dir": temp, "socket_ifname": "eth0", "model_timeout": 3600}
            original = {"VLLM_DP_SIZE": "2", "VLLM_HUMMING_MOE_GEMM_TYPE": "grouped", "RAY_ADDRESS": "stale",
                        "VLLM_USE_RUST_FRONTEND": "0", "CUDA_HOME": "/kit", "VLLM_CACHE_ROOT": "/cache",
                        "LD_LIBRARY_PATH": "/opt/rdma/lib", "NCCL_IB_HCA": "mlx5"}
            for rank in (0, 1):
                env = native_environment(config, rank, f"10.1.0.{rank+1}", original)
                for name in ("VLLM_DP_SIZE", "VLLM_HUMMING_MOE_GEMM_TYPE", "RAY_ADDRESS"):
                    self.assertNotIn(name, env)
                self.assertEqual(env["VLLM_USE_RUST_FRONTEND"], "1")
                self.assertEqual(env["VLLM_HOST_IP"], f"10.1.0.{rank+1}")
                self.assertEqual(env["NCCL_IB_HCA"], "mlx5")
                self.assertEqual(env["CUDA_HOME"], "/kit")
                self.assertEqual(env["VLLM_CACHE_ROOT"], "/cache")
                self.assertEqual("VLLM_API_KEY" in env, rank == 0)

    def test_bad_physical_placement_and_dependency_mismatch_rejected(self):
        left, right = metadata(0), metadata(1)
        validate_peers(left, right)
        for key, value, message in (("boot_hash", left["boot_hash"], "physical host"),
                                    ("packages_sha256", "different", "dependencies differ")):
            with self.assertRaisesRegex(RuntimeError, message):
                validate_peers(left, dict(right, **{key: value}))
        with self.assertRaisesRegex(RuntimeError, "UUIDs overlap"):
            validate_peers(left, dict(right, devices=left["devices"]))

    def test_early_process_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
            process.wait()
            with self.assertRaisesRegex(RuntimeError, "status 7"):
                wait_api(process, "http://unused", "key", "model", time.monotonic()+10, Path(temp)/"STOP")

    def test_headless_node_never_waits_for_or_calls_a_local_api(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            checkpoint(model)
            run, _ = prepare(arguments(["--model-dir", str(model), "--nfs-root", temp, "--prepare-only"]))
            atomic_json(run / "node-0.json", metadata(0))
            atomic_json(run / "head.json", {"ip": "10.1.0.1"})
            head_ready = {"base_url": "http://10.1.0.1:8000", "expires_at": time.time() + 60}
            atomic_json(run / "READY.json", head_ready)
            fake = run / "fake-worker.py"
            fake.write_text('import os,time\nfrom pathlib import Path\nPath(__file__).with_suffix(".pid").write_text(str(os.getpid()))\ntime.sleep(60)\n')

            def watcher():
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and not fake.with_suffix(".pid").exists():
                    time.sleep(0.05)
                (run / "STOP").touch()

            watch = threading.Thread(target=watcher)
            previous = signal.getsignal(signal.SIGTERM)
            watch.start()
            try:
                with patch("vllm_node.preflight", return_value=metadata(1)), \
                     patch("vllm_node.server_command", return_value=[sys.executable, str(fake)]), \
                     patch("vllm_node.api_request") as request, \
                     patch("vllm_node.wait_api") as api_wait:
                    serve_node(run, 1)
                    request.assert_not_called()
                    api_wait.assert_not_called()
            finally:
                signal.signal(signal.SIGTERM, previous)
                watch.join()
            self.assertEqual(read_json(run / "READY.json"), head_ready)
            self.assertEqual(read_json(run / "stopped-1.json")["reason"], "stop-requested")
            with self.assertRaises(ProcessLookupError):
                os.kill(int(fake.with_suffix(".pid").read_text()), 0)

    def test_diagnostics_include_worker_failure_without_api_key(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "api-key").write_text("secret")
            (run / "vllm-worker.log").write_text("ROOT CAUSE secret\n")
            report = diagnostic_report(run)
            self.assertIn("ROOT CAUSE", report)
            self.assertNotIn("secret", report)

    def test_ready_requires_real_child_api_generation_and_ib_evidence(self):
        # Uses a real child process and STOP lifecycle; HTTP and GPU imports are stubbed.
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            checkpoint(model)
            run, _ = prepare(arguments(["--model-dir", str(model), "--nfs-root", temp, "--prepare-only"]))
            atomic_json(run / "node-1.json", metadata(1))
            (run / "nccl.fake.log").write_text("Channel via NET/IB/0/GDRDMA\n")
            fake = run / "fake.py"
            fake.write_text('import os,time\nfrom pathlib import Path\nPath(__file__).with_suffix(".pid").write_text(str(os.getpid()))\ntime.sleep(60)\n')
            observations = []
            def request(base, path, key, payload=None, timeout=30):
                if not fake.with_suffix(".pid").exists():
                    raise OSError("Not started")
                if path == "/v1/models":
                    return {"data": [{"id": "deepseek-v4.1-flash"}]}
                if path == "/v1/chat/completions":
                    self.assertFalse((run / "READY.json").exists())
                    return {"choices": [{"message": {"content": "4"}}], "usage": {"completion_tokens": 1}}
                return {}
            def watcher():
                until = time.monotonic() + 15
                while time.monotonic() < until:
                    ready = read_json(run / "READY.json")
                    if ready:
                        observations.append(ready)
                        break
                    time.sleep(0.1)
                (run / "STOP").touch()
            watch = threading.Thread(target=watcher)
            previous = signal.getsignal(signal.SIGTERM)
            watch.start()
            try:
                with patch("vllm_node.preflight", return_value=metadata(0)), \
                     patch("vllm_node.api_request", side_effect=request), \
                     patch("vllm_node.server_command", return_value=[sys.executable, str(fake)]):
                    serve_node(run, 0)
            finally:
                signal.signal(signal.SIGTERM, previous)
                watch.join()
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0]["tensor_parallel_size"], 1)
            self.assertEqual(observations[0]["pipeline_parallel_size"], 1)
            self.assertEqual(observations[0]["frontend"], "rust")
            self.assertTrue(observations[0]["vllm_nccl_ib_channel_verified"])
            self.assertTrue(observations[0]["enable_expert_parallel"])
            self.assertEqual(observations[0]["data_parallel_size"], 16)
            self.assertEqual(observations[0]["load_balancing"], "internal")
            self.assertEqual(observations[0]["endpoints"], ["http://10.1.0.1:8000"])
            self.assertNotIn((run / "api-key").read_text().strip(), json.dumps(observations[0]))
            self.assertEqual(read_json(run / "stopped-0.json")["reason"], "stop-requested")
            with self.assertRaises(ProcessLookupError):
                os.kill(int(fake.with_suffix(".pid").read_text()), 0)


if __name__ == "__main__":
    unittest.main()
