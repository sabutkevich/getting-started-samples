"""Preserve GPU failure evidence before job cleanup, without requiring CUDA."""
import contextlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import ray_node


class Matrix:
    def __matmul__(self, other):
        return self

    def __eq__(self, other):
        return True


class GPUPreflightTests(unittest.TestCase):
    def fake_modules(self, busy_index=None):
        torch = types.SimpleNamespace(__version__="torch-test", version=types.SimpleNamespace(cuda="13.0"))
        cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 8,
                                     synchronize=lambda index: None,
                                     get_device_properties=lambda index: types.SimpleNamespace(name="H200", uuid=f"gpu-{index}"),
                                     nccl=types.SimpleNamespace(version=lambda: (2, 29, 7)))
        @contextlib.contextmanager
        def device(index):
            if index == busy_index:
                raise RuntimeError("CUDA-capable device(s) is/are busy or unavailable")
            yield
        cuda.device = device
        torch.cuda = cuda
        torch.ones = lambda *args, **kwargs: Matrix()
        torch.all = lambda value: types.SimpleNamespace(item=lambda: value)
        return {"torch": torch, "ray": types.SimpleNamespace(__version__=ray_node.RAY_VERSION)}

    def run_check(self, run, modules, output):
        with patch.dict(sys.modules, modules), \
             patch("ray_node.subprocess.run", side_effect=output), \
             patch("ray_node.pod_ipv4", return_value="10.1.0.1"), \
             patch("ray_node.fingerprint", return_value="boot-test"), \
             patch("ray_node.socket.gethostname", return_value="mpimaster-0"):
            return ray_node.preflight({"run_dir": str(run), "gpus_per_node": 8}, 0)

    def test_busy_device_keeps_index_identity_successful_devices_and_process_evidence(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-A,GPU-B",
                                                                          "API_KEY": "do-not-save"}):
            run = Path(temp)
            def output(command, **kwargs):
                snapshot = run / "gpu-preflight-0.json"
                if snapshot.exists():
                    saved = json.loads(snapshot.read_text())
                    self.assertEqual(saved["rank"], 0)
                    self.assertEqual(saved["current_device_index"], 3)
                return types.SimpleNamespace(returncode=0, stdout="GPU-A, Exclusive_Process, existing-process\n", stderr="")
            with self.assertRaisesRegex(RuntimeError, "rank 0, pod mpimaster-0, CUDA device 3") as caught:
                self.run_check(run, self.fake_modules(busy_index=3), output)
            saved = json.loads((run / "gpu-preflight-0.json").read_text())
            self.assertEqual(saved["status"], "FAILED")
            self.assertEqual(saved["failed_device_index"], 3)
            self.assertEqual(saved["failed_stage"], "device-context")
            self.assertEqual([entry["index"] for entry in saved["checks"]], [0, 1, 2])
            self.assertEqual(saved["environment"]["CUDA_VISIBLE_DEVICES"], "GPU-A,GPU-B")
            self.assertIn("existing-process", saved["at_failure"]["compute_processes"]["stdout"])
            self.assertNotIn("do-not-save", json.dumps(saved))
            self.assertIsNotNone(caught.exception.__cause__)

    def test_missing_nvidia_smi_does_not_hide_the_cuda_error(self):
        with tempfile.TemporaryDirectory() as temp:
            def missing(*args, **kwargs):
                raise FileNotFoundError("nvidia-smi unavailable")
            with self.assertRaisesRegex(RuntimeError, "CUDA device 0.*busy or unavailable"):
                self.run_check(Path(temp), self.fake_modules(busy_index=0), missing)
            saved = json.loads((Path(temp) / "gpu-preflight-0.json").read_text())
            self.assertIn("error", saved["before_cuda"]["gpus"])
            self.assertIn("error", saved["at_failure"]["compute_processes"])

    def test_success_still_checks_all_eight_requested_devices(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            def output(command, **kwargs):
                calls.append(command)
                return types.SimpleNamespace(returncode=0, stdout="healthy", stderr="")
            class Port:
                def __str__(self):
                    return "/sys/class/infiniband/mlx5_0/ports/1"

                def __truediv__(self, key):
                    values = {"state": "4: ACTIVE", "rate": "200 Gb/sec", "link_layer": "InfiniBand"}
                    return types.SimpleNamespace(read_text=lambda: values[key])
            with patch("ray_node.Path.glob", return_value=[Port()]):
                info = self.run_check(Path(temp), self.fake_modules(), output)
            self.assertEqual(info["boot_hash"], "boot-test")
            self.assertEqual(info["rdma_ports"][0]["state"], "4: ACTIVE")
            saved = json.loads((Path(temp) / "gpu-preflight-0.json").read_text())
            self.assertEqual(saved["status"], "PASSED")
            self.assertEqual(len(saved["checks"]), 8)
            self.assertEqual([device["uuid"] for device in info["devices"]], [f"gpu-{i}" for i in range(8)])
            self.assertEqual(len(calls), 2)
            self.assertEqual(info["gpu_preflight_file"], str(Path(temp) / "gpu-preflight-0.json"))

    def test_wrong_visible_count_fails_before_accessing_any_device(self):
        modules = self.fake_modules()
        modules["torch"].cuda.device_count = lambda: 2
        with tempfile.TemporaryDirectory() as temp:
            def output(command, **kwargs):
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")
            with self.assertRaisesRegex(RuntimeError, "Expected 8 visible GPU"):
                self.run_check(Path(temp), modules, output)
            saved = json.loads((Path(temp) / "gpu-preflight-0.json").read_text())
            self.assertIsNone(saved["failed_device_index"])
            self.assertEqual(saved["checks"], [])
