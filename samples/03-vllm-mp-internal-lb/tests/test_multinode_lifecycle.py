"""Exercise API ownership and readiness across four automatically discovered workers."""
import contextlib
import io
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cluster_common import atomic_json, read_json, wait_json
from submit_vllm import arguments, prepare
from test_dep import checkpoint, metadata
from vllm_common import server_command
from vllm_node import serve_node

HYBRID = Path(__file__).resolve().parents[1].name.startswith("04")


class MultiNodeLifecycleTests(unittest.TestCase):
    def run_role(self, rank):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            checkpoint(model)
            run, _ = prepare(arguments(["--model-dir", str(model), "--nfs-root", temp,
                                         "--prepare-only", "--num-nodes", "4", "--tp", "8", "--pp", "2"]))
            for node in range(4):
                atomic_json(run / f"node-{node}.json", metadata(node))
            atomic_json(run / "head.json", {"ip": "10.1.0.1"})
            (run / "nccl.fake.log").write_text("Channel via NET/IB/0/GDRDMA\n")
            config = read_json(run / "run.json")
            expected_apis = [0, 2] if HYBRID else [0]
            captured = []
            requests = []
            observations = []
            if rank:
                atomic_json(run / "READY.json", {"expires_at": time.time() + 60})

            class Process:
                def poll(self):
                    if (run / "READY.json").exists():
                        (run / "STOP").touch()
                    return None

            def command(runtime, node_rank, head_ip):
                result = server_command(runtime, node_rank, head_ip)
                captured.append(result)
                return result

            def request(base, path, key, payload=None, timeout=30):
                requests.append(base)
                return {"choices": [{"message": {"content": "4"}}], "usage": {"completion_tokens": 1}}

            def wait_for(path, timeout, stop):
                if path.name == "api-ready-2.json":
                    # The head must wait for the API on the second replica leader.
                    self.assertTrue((run / "api-ready-0.json").exists())
                    self.assertFalse((run / "READY.json").exists())
                    atomic_json(path, {"base_url": "http://10.1.0.3:8000"})
                    observations.append("second-replica-ready")
                return wait_json(path, timeout, stop)

            previous = signal.getsignal(signal.SIGTERM)
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                     patch("vllm_node.preflight", return_value=metadata(rank)), \
                     patch("vllm_node.server_command", side_effect=command), \
                     patch("vllm_node.subprocess.Popen", return_value=Process()), \
                     patch("vllm_node.wait_api") as wait_api, \
                     patch("vllm_node.api_request", side_effect=request), \
                     patch("vllm_node.wait_json", side_effect=wait_for), \
                     patch("vllm_node.stop_process_group") as cleanup, \
                     patch("vllm_node.time.sleep"):
                    serve_node(run, rank)
            finally:
                signal.signal(signal.SIGTERM, previous)

            self.assertEqual(len(captured), 1)
            command = captured[0]
            self.assertEqual(command[command.index("--master-addr") + 1],
                             f"10.1.0.{(rank // 2) * 2 + 1}")
            if rank in expected_apis:
                wait_api.assert_called_once()
                self.assertEqual(requests, [f"http://10.1.0.{rank+1}:8000"])
                local = read_json(run / f"api-ready-{rank}.json")
                self.assertEqual(local["data_parallel_start_rank"], rank // 2)
                self.assertNotIn("--headless", command)
            else:
                wait_api.assert_not_called()
                self.assertEqual(requests, [])
                self.assertIn("--headless", command)
                self.assertFalse((run / f"api-ready-{rank}.json").exists())
            if rank == 0:
                ready = read_json(run / "READY.json")
                self.assertEqual(ready["endpoint_nodes"], expected_apis)
                self.assertEqual(ready["physical_nodes"], [f"boot-{r}" for r in range(4)])
                self.assertEqual(ready["gpus"], 32)
                self.assertEqual(observations, ["second-replica-ready"] if HYBRID else [])
            cleanup.assert_called_once()
            self.assertEqual(read_json(run / f"stopped-{rank}.json")["rank"], rank)

    def test_head_waits_for_all_api_leaders_before_readiness(self):
        self.run_role(0)

    def test_followers_on_both_replicas_have_no_local_api(self):
        for rank in (1, 3):
            with self.subTest(rank=rank):
                self.run_role(rank)

    def test_second_replica_leader_owns_only_its_api_and_rendezvous(self):
        self.run_role(2)
