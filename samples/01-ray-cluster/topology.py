"""Resource and parallelism planning without GPU or platform SDK imports."""
import argparse
import time
from pathlib import Path


def add_cluster_options(parser):
    parser.add_argument("--num-nodes", type=int, default=2, help="Number of automatically placed worker nodes")
    parser.add_argument("--allocation-name", help="Cloud.ru training allocation; omitted: workspace default")
    parser.add_argument("--queue-name", default="shared", help="Cloud.ru queue; default: %(default)s")


def add_parallel_options(parser, default_tp):
    add_cluster_options(parser)
    parser.add_argument("--gpus-per-node", type=int, default=8, help="Identical H200 allocation on each worker, 1..8")
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=default_tp)
    parser.add_argument("--pipeline-parallel-size", "--pp", type=int, default=1)
    parser.add_argument("--data-parallel-size", "--dp", type=int, help="Defaults to all allocated GPUs / (TP * PP)")
    parser.add_argument("--data-parallel-size-local", type=int, help="Optional assertion of the calculated local DP count")
    parser.add_argument("--enable-expert-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expert-parallel-size", "--ep-size", type=int,
                        help="Optional assertion: EP=TP*DP when enabled, otherwise 1")


def resource_values(args):
    return {"num_nodes": args.num_nodes, "gpus_per_node": args.gpus_per_node,
            "allocation_name": args.allocation_name, "queue_name": args.queue_name}


def plan_topology(nodes, gpus, tp, dp=None, pp=1, ep=True, expected_ep=None, expected_local=None):
    for name, value in (("num_nodes", nodes), ("gpus_per_node", gpus),
                        ("tensor_parallel_size", tp), ("pipeline_parallel_size", pp)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if gpus > 8:
        raise ValueError("H200 worker shapes support 1..8 GPUs per node")
    total, replica = nodes * gpus, tp * pp
    if total % replica:
        raise ValueError(f"Allocated GPUs ({total}) must be divisible by TP*PP ({replica})")
    dp = total // replica if dp is None else dp
    if type(dp) is not int or dp <= 0 or dp * replica != total:
        raise ValueError(f"DP*TP*PP must equal num_nodes*gpus_per_node ({total}); got {dp}*{tp}*{pp}")
    if replica <= gpus:
        if gpus % replica:
            raise ValueError("TP*PP must divide GPUs per node for uniform local DP placement")
        local, span = gpus // replica, 1
    else:
        if replica % gpus:
            raise ValueError("A spanning TP*PP replica must use a whole number of worker nodes")
        local, span = 1, replica // gpus
    ep_size = tp * dp if ep else 1
    if expected_ep is not None and expected_ep != ep_size:
        raise ValueError(f"Requested EP={expected_ep}, but vLLM derives EP={ep_size} from TP*DP and the EP flag")
    if expected_local is not None and expected_local != local:
        raise ValueError(f"Requested local DP={expected_local}, but this topology requires {local}")
    return {"num_nodes": nodes, "gpus_per_node": gpus, "total_gpus": total,
            "tensor_parallel_size": tp, "data_parallel_size": dp, "pipeline_parallel_size": pp,
            "enable_expert_parallel": ep, "expert_parallel_size": ep_size,
            "data_parallel_size_local": local, "nodes_per_replica": span}


def topology_from_args(args):
    return plan_topology(args.num_nodes, args.gpus_per_node, args.tensor_parallel_size,
                         args.data_parallel_size, args.pipeline_parallel_size, args.enable_expert_parallel,
                         args.expert_parallel_size, args.data_parallel_size_local)


def resolved_topology(config, default_tp=1):
    return plan_topology(config.get("num_nodes", 2), config.get("gpus_per_node", 8),
                         config.get("tensor_parallel_size", default_tp), config.get("data_parallel_size"),
                         config.get("pipeline_parallel_size", 1), config.get("enable_expert_parallel", True),
                         config.get("expert_parallel_size"), config.get("data_parallel_size_local"))


def api_nodes(config):
    if config.get("load_balancing", "internal") != "hybrid":
        return [0]
    topo = resolved_topology(config)
    return list(range(0, topo["num_nodes"], topo["nodes_per_replica"]))


def validate_geometry(manifest, config, default_tp=1):
    topo = resolved_topology(config, default_tp)
    tp = topo["tensor_parallel_size"]
    geometry = manifest.get("text_geometry", {})
    for key in ("o_groups", "num_attention_heads"):
        value = geometry.get(key)
        if value is not None and (type(value) is not int or value < tp or value % tp):
            raise ValueError(f"Checkpoint {key}={value} is incompatible with TP={tp}")
    # Uneven expert placement is supported by vLLM. Do not require experts % EP == 0.
    return topo


def validate_nodes(metadata, count):
    if sorted(entry["rank"] for entry in metadata) != list(range(count)):
        raise RuntimeError("Worker metadata must contain every configured node rank exactly once")
    for key, message in (("ip", "Workers share a pod IP"),
                         ("boot_hash", "Workers share a physical host")):
        values = [entry.get(key) for entry in metadata if entry.get(key)]
        if len(values) != len(set(values)):
            raise RuntimeError(message)
    for key in ("ray", "torch", "vllm", "packages_sha256"):
        values = {str(entry[key]) for entry in metadata if key in entry}
        if len(values) > 1:
            raise RuntimeError(f"Installed dependencies differ between nodes: {key}")
    uuids = [device["uuid"] for entry in metadata for device in entry.get("devices", [])
             if device.get("uuid") not in (None, "unknown")]
    if len(uuids) != len(set(uuids)):
        raise RuntimeError("GPU UUIDs overlap between processes/nodes")


def wait_nodes(run, config, stop):
    from cluster_common import wait_json
    deadline = time.monotonic() + config["startup_timeout"]
    metadata = [wait_json(Path(run) / f"node-{rank}.json", max(.001, deadline-time.monotonic()), stop)
                for rank in range(config["num_nodes"])]
    validate_nodes(metadata, config["num_nodes"])
    return metadata


def submit_checked(job, run):
    from cluster_common import atomic_json
    reply = job.submit()
    name = getattr(job, "job_name", None)
    print(f"Submission response: {reply}", flush=True)
    atomic_json(Path(run) / "platform-job.json", {"job_name": name, "reply": reply})
    if not isinstance(name, str) or not name or name == "None" or reply != f'Job "{name}" created.':
        raise RuntimeError(f"Platform did not create the job: {reply}")
    print(f"Platform job: {name}; {job.status()}", flush=True)
    return name
