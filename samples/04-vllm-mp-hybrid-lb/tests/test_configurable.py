"""Check non-default inference topology plans and role-specific commands."""
import contextlib
import io
import json
import unittest
from pathlib import Path

from cluster_common import job_plan
from submit_vllm import arguments, prepare
from topology import api_nodes, plan_topology
from vllm_common import server_command

RAY = Path(__file__).resolve().parents[1].name.startswith("02")


class ConfigurableInferenceTests(unittest.TestCase):
    def dry_plan(self, *options):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            prepare(arguments(["--dry-run", *options]))
        return json.loads(output.getvalue())

    def test_ten_nodes_have_automatic_dp_and_one_platform_job(self):
        result = self.dry_plan("--num-nodes", "10", "--tp", "1",
                               "--allocation-name", "alloc", "--queue-name", "shared")
        config = result["configuration"]
        self.assertEqual((config["data_parallel_size"], config["expert_parallel_size"]), (80, 80))
        self.assertEqual(result["job_requests"][0]["n_workers"], 10)
        self.assertNotIn("include_nodes", result["job_requests"][0])
        if not RAY:
            self.assertEqual(len(result["server_commands"]), 10)

    def test_explicit_tp_dp_and_ep_disable_reach_the_runtime_command(self):
        result = self.dry_plan("--num-nodes", "4", "--gpus-per-node", "8",
                               "--tp", "4", "--dp", "8", "--no-enable-expert-parallel", "--ep-size", "1")
        config = result["configuration"]
        command = result["server_command"] if RAY else result["server_commands"][0]
        self.assertEqual(config["expert_parallel_size"], 1)
        self.assertNotIn("--enable-expert-parallel", command)
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "4")
        self.assertEqual(command[command.index("--data-parallel-size-local") + 1], "2")

    def test_single_node_is_supported_without_a_remote_api(self):
        result = self.dry_plan("--num-nodes", "1", "--tp", "1")
        self.assertEqual(result["configuration"]["total_gpus"], 8)
        if not RAY:
            command = result["server_commands"][0]
            self.assertNotIn("--headless", command)
            self.assertNotIn("--data-parallel-hybrid-lb", command)

    def test_replicas_spanning_nodes_use_correct_execution_mode(self):
        result = self.dry_plan("--num-nodes", "4", "--tp", "8", "--pp", "2")
        config = result["configuration"]
        self.assertEqual(config["nodes_per_replica"], 2)
        self.assertEqual(config["data_parallel_size"], 2)
        if RAY:
            self.assertEqual(result["server_command"][result["server_command"].index("--data-parallel-size-local")+1], "1")
        else:
            for rank, command in enumerate(result["server_commands"]):
                self.assertEqual(command[command.index("--nnodes")+1], "4")
                self.assertEqual(command[command.index("--node-rank")+1], str(rank))
                self.assertEqual(command[command.index("--pipeline-parallel-size")+1], "2")
                if rank % 2:
                    self.assertIn("--headless", command)
                    self.assertNotIn("--data-parallel-hybrid-lb", command)

    def test_non_head_native_ranks_have_disjoint_dp_ownership(self):
        if RAY:
            self.skipTest("Ray owns engine placement")
        result = self.dry_plan("--num-nodes", "5", "--tp", "2")
        for rank, command in enumerate(result["server_commands"]):
            if rank:
                self.assertEqual(command[command.index("--data-parallel-start-rank")+1], str(rank * 4))
        if result["configuration"]["load_balancing"] == "internal":
            self.assertNotIn("--data-parallel-start-rank", result["server_commands"][0])
            self.assertTrue(all("--headless" in c for c in result["server_commands"][1:]))
        else:
            self.assertTrue(all("--headless" not in c for c in result["server_commands"]))

    def test_invalid_flags_fail_without_model_reads_or_platform_submission(self):
        for options in (["--num-nodes", "0"], ["--num-nodes", "3", "--tp", "16"],
                        ["--num-nodes", "4", "--dp", "3"], ["--ep-size", "3"],
                        ["--data-parallel-size-local", "9"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(options)
