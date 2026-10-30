"""Offline checks for placement, bootstrap generation and two-rank bandwidth reporting."""
import json
import shlex
import tempfile
import unittest
from pathlib import Path

from cluster_common import BASE_IMAGE, SOURCE_FILES, atomic_json, job_plan, nccl_bus_bandwidth, read_json, resolve_rank
from ray_node import ray_command
from submit_ray import arguments, prepare, request_stop


class ClusterTests(unittest.TestCase):
    def config(self, **changes):
        config = {"id": "abc123", "image": BASE_IMAGE, "nodes": ["node-a", "node-b"],
                  "gpus_per_node": 1, "cpus_per_node": 12, "placement": "pinned",
                  "object_store_bytes": 1024**3}
        config.update(changes)
        return config

    def test_one_gpu_jobs_cannot_pack_onto_one_allowed_node(self):
        jobs = job_plan(self.config())
        self.assertEqual(len(jobs), 2)
        self.assertEqual([job["include_nodes"] for job in jobs], [["node-a"], ["node-b"]])
        self.assertTrue(all(job["n_workers"] == job["processes_per_worker"] == 1 for job in jobs))
        self.assertTrue(all(job["instance_type"] == "h200.1gpu.12C.244G" for job in jobs))
        with self.assertRaises(ValueError):
            job_plan(self.config(nodes=["node-a", "node-a"]))

    def test_full_node_configuration_requires_explicit_gpu_request(self):
        jobs = job_plan(self.config(gpus_per_node=8, cpus_per_node=96))
        self.assertTrue(all(job["instance_type"] == "h200.8gpu.96C.1952G" for job in jobs))
        self.assertEqual(len(jobs), 2)

    def test_binary_mpi_rank_takes_precedence_over_stale_torch_rank(self):
        self.assertEqual(resolve_rank("auto", {"OMPI_COMM_WORLD_RANK": "1", "RANK": "0"}), 1)
        self.assertEqual(resolve_rank("1", {"RANK": "0"}), 1)
        with self.assertRaises(RuntimeError):
            resolve_rank("auto", {})

    def test_bandwidth_matches_nccl_formula_without_duplex_inflation(self):
        # 1 GiB per operation, 32 operations in 2 seconds.
        self.assertAlmostEqual(nccl_bus_bandwidth(1024**3, 32, 2, 2), 17.179869184)
        self.assertAlmostEqual(nccl_bus_bandwidth(1024**3, 32, 2, 4), 25.769803776)

    def test_ray_head_and_worker_explicit_ips_and_ports(self):
        head = ray_command(self.config(), 0, "10.0.0.1")
        worker = ray_command(self.config(), 1, "10.0.0.2", "10.0.0.1:6379")
        self.assertIn("--head", head)
        self.assertNotIn("--head", worker)
        self.assertIn("--address=10.0.0.1:6379", worker)
        self.assertIn("--node-ip-address=10.0.0.2", worker)
        self.assertIn("--block", head)
        self.assertIn("--block", worker)
        for name in ("node-manager", "object-manager", "runtime-env-agent", "dashboard-agent-grpc"):
            self.assertTrue(any(arg.startswith(f"--{name}-port=") for arg in head))

    def test_prepared_bootstrap_handles_nfs_path_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="cloud ray test ") as temp:
            args = arguments(["--nfs-root", temp, "--prepare-only"])
            run, jobs = prepare(args)
            self.assertEqual(len(jobs), 2)
            config = read_json(run / "run.json")
            self.assertEqual(config["keep_seconds"], 0)
            self.assertEqual(config["image"], BASE_IMAGE)
            for name in SOURCE_FILES:
                self.assertTrue((run / name).is_file())
            for rank, job in enumerate(jobs):
                command = Path(job["script"]).read_text().splitlines()[-1]
                parts = shlex.split(command)
                self.assertEqual(parts[:3], ["exec", "python", "-u"])
                self.assertEqual(Path(parts[3]), run / "bootstrap_ray.py")
                self.assertEqual(parts[-1], str(rank))
            request_stop(run)
            self.assertTrue((run / "STOP").exists())

    def test_single_job_has_auto_rank_and_two_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(["--nfs-root", temp, "--prepare-only", "--placement", "single-job"])
            run, jobs = prepare(args)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["n_workers"], 2)
            self.assertEqual(shlex.split(Path(jobs[0]["script"]).read_text().splitlines()[-1])[-1], "auto")

    def test_keep_alive_is_bounded_and_atomic_files_are_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(["--nfs-root", temp, "--prepare-only", "--keep-alive", "--keep-minutes", "20"])
            run, _ = prepare(args)
            self.assertEqual(read_json(run / "run.json")["keep_seconds"], 1200)
            path = run / "status.json"
            atomic_json(path, {"state": "ready"})
            self.assertEqual(json.loads(path.read_text()), {"state": "ready"})
            self.assertEqual(list(run.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
