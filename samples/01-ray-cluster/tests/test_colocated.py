"""Reproduce the reported three-worker/two-GPU placement without Ray or cloud GPUs."""
import contextlib
import io
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cluster_common import atomic_json, read_json
from ray_node import run_node
from submit_ray import arguments, prepare


class ColocatedWorkerLifecycleTests(unittest.TestCase):
    def test_each_colocated_worker_passes_preflight_and_reaches_ray_startup(self):
        for rank in range(3):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as temp:
                run, _ = prepare(arguments(["--num-nodes", "3", "--gpus-per-node", "2",
                                             "--prepare-only", "--nfs-root", temp]))
                nodes = [{"rank": node, "ip": f"10.1.0.{node+1}", "boot_hash": "server-218",
                          "ray": "same", "torch": "same",
                          "devices": [{"uuid": f"gpu-{node}-{index}"} for index in range(2)]}
                         for node in range(3)]
                for node in nodes:
                    atomic_json(run / f"node-{node['rank']}.json", node)
                atomic_json(run / "head.json", {"address": "10.1.0.1:6379"})
                # Let the supervisor finish immediately after startup, without real GPU work.
                atomic_json(run / "PASSED.json", {"completed_at": time.time()})
                ray_process = Mock()
                ray_process.poll.return_value = None
                smoke_process = Mock(returncode=0)
                smoke_process.poll.return_value = 0

                def start(command, **kwargs):
                    return smoke_process if any(str(part).endswith("smoke_test.py") for part in command) else ray_process

                previous = signal.getsignal(signal.SIGTERM)
                try:
                    with contextlib.redirect_stdout(io.StringIO()), \
                         patch("ray_node.preflight", return_value=nodes[rank]), \
                         patch("ray_node.subprocess.Popen", side_effect=start) as popen, \
                         patch("ray_node.subprocess.run"), \
                         patch("ray_node.socket.create_connection"), \
                         patch("ray_node.archive_logs"):
                        run_node(run, rank)
                finally:
                    signal.signal(signal.SIGTERM, previous)
                command = popen.call_args_list[0].args[0]
                self.assertEqual(command[:2], ["ray", "start"])
                self.assertIn(f"--node-ip-address=10.1.0.{rank+1}", command)
                self.assertIn("--num-gpus=2", command)
                self.assertEqual(read_json(run / f"stopped-{rank}.json")["rank"], rank)
                self.assertFalse(list(run.glob("*-error*.json")))
