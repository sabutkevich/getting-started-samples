"""Offline topology, rank, node-discovery and platform-submission regressions."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cluster_common import BASE_IMAGE, atomic_json, job_plan, resolve_rank
from topology import (api_nodes, plan_topology, resolved_topology, submit_checked,
                      validate_geometry, validate_nodes, wait_nodes, placement_summary, requires_interhost_ib)


class TopologyTests(unittest.TestCase):
    def test_arbitrary_node_counts_are_requested_in_one_job_without_host_pinning(self):
        for count in (1, 2, 3, 5, 10, 32):
            config = {"num_nodes": count, "gpus_per_node": 8, "cpus_per_node": 96,
                      "image": BASE_IMAGE, "allocation_name": "allocation", "queue_name": "shared"}
            plan = job_plan(config)
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0]["n_workers"], count)
            self.assertEqual(plan[0]["allocation_name"], "allocation")
            self.assertEqual(plan[0]["queue_name"], "shared")
            self.assertNotIn("include_nodes", plan[0])
            self.assertNotIn("exclude_nodes", plan[0])

    def test_rank_discovery_accepts_all_configured_workers_and_checks_bounds(self):
        self.assertEqual(resolve_rank("auto", {"OMPI_COMM_WORLD_RANK": "9", "RANK": "0"}, 10), 9)
        self.assertEqual(resolve_rank("auto", {}, 1), 0)
        for rank in ("-1", "10"):
            with self.assertRaises(ValueError):
                resolve_rank(rank, {}, 10)

    def test_valid_topology_matrix_uses_every_allocated_gpu(self):
        valid = 0
        for nodes in (1, 2, 3, 4, 5, 10):
            for gpus in (1, 2, 4, 8):
                for tp in (1, 2, 4, 8, 16):
                    for pp in (1, 2):
                        replica = tp * pp
                        if nodes * gpus % replica:
                            continue
                        if replica <= gpus and gpus % replica:
                            continue
                        if replica > gpus and replica % gpus:
                            continue
                        for ep in (False, True):
                            config = plan_topology(nodes, gpus, tp, pp=pp, ep=ep)
                            self.assertEqual(config["data_parallel_size"] * tp * pp, nodes * gpus)
                            self.assertEqual(config["expert_parallel_size"], tp * config["data_parallel_size"] if ep else 1)
                            self.assertEqual(config["nodes_per_replica"] * gpus,
                                             replica if replica > gpus else gpus)
                            self.assertEqual(resolved_topology(config), config)
                            valid += 1
        self.assertGreater(valid, 100)

    def test_invalid_parallelism_is_rejected_before_allocation(self):
        for values in ((0, 8, 1), (2, 9, 1), (2, 8, 0), (2, 8, 3),
                       (3, 8, 16), (2, 8, 1, 2)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                plan_topology(*values)
        with self.assertRaisesRegex(ValueError, "derives EP"):
            plan_topology(2, 8, 2, expected_ep=4)
        with self.assertRaisesRegex(ValueError, "local DP"):
            plan_topology(2, 8, 2, expected_local=8)

    def test_hybrid_api_leaders_follow_replica_boundaries(self):
        self.assertEqual(api_nodes(dict(plan_topology(10, 8, 1), load_balancing="hybrid")), list(range(10)))
        self.assertEqual(api_nodes(dict(plan_topology(4, 8, 8, pp=2), load_balancing="hybrid")), [0, 2])
        self.assertEqual(api_nodes(dict(plan_topology(4, 8, 8, pp=2), load_balancing="internal")), [0])

    def test_model_geometry_checks_tp_without_rejecting_uneven_expert_placement(self):
        manifest = {"text_geometry": {"o_groups": 8, "num_attention_heads": 64, "n_routed_experts": 384}}
        validate_geometry(manifest, plan_topology(10, 8, 1))
        with self.assertRaisesRegex(ValueError, "o_groups=8"):
            validate_geometry(manifest, plan_topology(2, 8, 16))

    def test_all_nodes_are_discovered_in_rank_order_and_colocation_is_allowed(self):
        metadata = [{"rank": rank, "ip": f"10.1.0.{rank+1}", "boot_hash": f"boot-{rank}",
                     "devices": [{"uuid": f"gpu-{rank}"}], "packages_sha256": "same"}
                    for rank in range(5)]
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for entry in reversed(metadata):
                atomic_json(run / f"node-{entry['rank']}.json", entry)
            self.assertEqual(wait_nodes(run, {"num_nodes": 5, "startup_timeout": 1}, run / "STOP"), metadata)
        validate_nodes(metadata, 5)
        validate_nodes([*metadata[:-1], dict(metadata[-1], boot_hash="boot-0")], 5)
        with self.assertRaisesRegex(RuntimeError, "pod IP"):
            validate_nodes([*metadata[:-1], dict(metadata[-1], ip=metadata[0]["ip"])], 5)
        with self.assertRaisesRegex(RuntimeError, "UUIDs overlap"):
            validate_nodes([*metadata[:-1], dict(metadata[-1], devices=metadata[0]["devices"])], 5)
        with self.assertRaisesRegex(RuntimeError, "every configured node"):
            validate_nodes(metadata[:-1], 5)

    def test_three_colocated_workers_have_six_distinct_gpus_and_one_physical_host(self):
        metadata = [{"rank": rank, "ip": f"10.1.0.{rank+1}", "boot_hash": "server-218",
                     "devices": [{"uuid": f"gpu-{rank}-{gpu}"} for gpu in range(2)]}
                    for rank in range(3)]
        validate_nodes(metadata, 3)
        self.assertEqual(placement_summary(metadata),
                         {"worker_count": 3, "physical_host_count": 1, "cross_physical_host": False})
        self.assertFalse(requires_interhost_ib(metadata))
        distributed = [*metadata[:-1], dict(metadata[-1], boot_hash="server-219")]
        self.assertEqual(placement_summary(distributed)["physical_host_count"], 2)
        self.assertTrue(requires_interhost_ib(distributed))
        unknown = [dict(entry, boot_hash=None) for entry in metadata]
        self.assertIsNone(placement_summary(unknown)["physical_host_count"])
        self.assertTrue(requires_interhost_ib(unknown))
        with self.assertRaisesRegex(RuntimeError, "UUIDs overlap"):
            validate_nodes([*metadata[:-1], dict(metadata[-1], devices=metadata[0]["devices"])], 3)

    def test_platform_string_error_is_not_treated_as_an_accepted_job(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            run = Path(temp)
            job = SimpleNamespace(job_name=None, submit=lambda: "Error 400: Resource allocation exceeds limit")
            with self.assertRaisesRegex(RuntimeError, "Platform did not create"):
                submit_checked(job, run)
            saved = json.loads((run / "platform-job.json").read_text())
            self.assertIsNone(saved["job_name"])
            job = SimpleNamespace(job_name="lm-mpi-job-test", submit=lambda: 'Job "lm-mpi-job-test" created.',
                                  status=lambda: "Job status=Pending")
            self.assertEqual(submit_checked(job, run), job.job_name)


if __name__ == "__main__":
    unittest.main()
