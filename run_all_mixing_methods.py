#!/usr/bin/env python
"""Run all mixing methods sequentially for comparison using the same topology.

Usage:
    # Use hardcoded defaults (backward-compatible):
    python run_all_mixing_methods.py

    # Run multiple experiments from a YAML file:
    python run_all_mixing_methods.py --experiments_yaml experiments.yaml
"""

import argparse
import subprocess
import sys
from datetime import datetime
import pickle
from pathlib import Path

import yaml

# src path — topology import is deferred inside resolve_topology() to avoid
# loading torch at import time (keeps --help instant)
sys.path.insert(0, str(Path(__file__).parent))

# ── Default configuration (used when no YAML is provided) ──────────────────
DEFAULTS = dict(
    num_clients       = 40,
    rounds            = 200,
    epochs            = 2,
    gossip_steps      = 1,
    gossip_schedule   = None,   # e.g. "5:0,3:100,1:200" — overrides gossip_steps
    delay_d           = 0,      # delayed aggregation depth d
    dataset           = "cifar10",
    model             = "resnet8",
    main_link_prob    = 1.0,
    border_link_prob  = 1.0,
    intra_cluster_prob= 0.8,
    topology_file     = None,   # auto-managed if None
    init_weights      = None,   # auto-managed if None
    partition_file    = None,   # auto-managed if None
    methods           = None,   # None → run all 5
)

ALL_METHODS = [
    ("metropolis_hastings",   "Metropolis-Hastings (Default)"),
    ("max_degree",            "Max-Degree (Uniform Weights)"),
    ("jaccard",               "Jaccard Similarity"),
    ("jaccard_dissimilarity", "Jaccard Dissimilarity (Avrachenkov)"),
    ("matcha",                "MATCHA (Optimal)"),
]


# ── Artefact helpers ────────────────────────────────────────────────────────

def resolve_topology(cfg: dict) -> Path:
    if cfg.get("topology_file"):
        path = Path(cfg["topology_file"])
        print(f"  topology : {path} (specified)")
        return path

    requested_n = int(cfg["num_clients"])
    exp_name = cfg.get("name", "default")
    path = Path(f"shared_topology_{exp_name}.pkl")

    def _generate(path_obj: Path):
        print(f"  topology : {path_obj} (generating ...)")
        from src.decentralized.topology import create_two_cluster_topology
        graph_obj = create_two_cluster_topology(
            num_clients=requested_n,
            main_link_prob=cfg["main_link_prob"],
            border_link_prob=cfg["border_link_prob"],
            intra_cluster_prob=cfg["intra_cluster_prob"],
            intra_cluster_communication=False,
        )
        with open(path_obj, "wb") as f:
            pickle.dump(graph_obj, f)
        print(f"    saved to {path_obj}")

    if path.exists():
        try:
            with open(path, "rb") as f:
                cached_graph = pickle.load(f)
            cached_nodes = set(cached_graph.nodes())
            expected_nodes = set(range(requested_n))
            if cached_nodes == expected_nodes:
                print(f"  topology : {path} (cached)")
                return path
            print(
                f"  topology : {path} (cached mismatch: {len(cached_nodes)} nodes, expected {requested_n})"
            )
            print("             regenerating to match requested client count ...")
            _generate(path)
        except Exception as e:
            print(f"  topology : {path} (cached read failed: {e})")
            print("             regenerating ...")
            _generate(path)
    else:
        _generate(path)
    return path


def resolve_init_weights(cfg: dict) -> Path:
    if cfg.get("init_weights"):
        path = Path(cfg["init_weights"])
        print(f"  weights  : {path} (specified)")
        return path
    path = Path("init_weights") / f"{cfg['model']}_w0.pt"
    if not path.exists():
        print(f"  weights  : {path} not found — running generate_init_weights.py ...")
        subprocess.run([sys.executable, "generate_init_weights.py"], check=True)
    print(f"  weights  : {path}")
    return path


def resolve_partition(cfg: dict) -> Path:
    if cfg.get("partition_file"):
        path = Path(cfg["partition_file"])
        print(f"  partition: {path} (specified)")
        return path
    n = cfg["num_clients"]
    ds = cfg["dataset"]
    partition_type = cfg.get("partition", "dirichlet")
    alpha = float(cfg.get("alpha", 0.5))
    classes_per_client = int(cfg.get("classes_per_client", 2))

    if partition_type == "pathological":
        fname = f"{ds}_N{n}_pathological_c{classes_per_client}.pkl"
    else:
        fname = f"{ds}_N{n}_{partition_type}_a{alpha}.pkl"

    path = Path("data_partition") / fname
    if not path.exists():
        print(f"  partition: {path} not found — generating requested partition ...")
        cmd = [
            sys.executable,
            "generate_partition.py",
            "--dataset", ds,
            "--num_clients", str(n),
            "--partition", partition_type,
            "--alpha", str(alpha),
            "--classes_per_client", str(classes_per_client),
        ]
        subprocess.run(cmd, check=True)
        if not path.exists():
            raise FileNotFoundError(
                f"generate_partition.py ran but {path} was not created.\n"
                f"Check generate_partition.py arguments and dataset availability."
            )
    print(f"  partition: {path}")
    return path


# ── Core runner ─────────────────────────────────────────────────────────────

def run_experiment(cfg: dict) -> list:
    """Run all (or a subset of) mixing methods for one experiment config."""
    exp_name      = cfg.get("name", "experiment")
    n_clients     = cfg["num_clients"]
    rounds        = cfg["rounds"]
    epochs        = cfg["epochs"]
    dataset       = cfg["dataset"]
    model         = cfg["model"]
    gossip_sched  = cfg.get("gossip_schedule")
    gossip_steps  = cfg.get("gossip_steps", 1)
    delay_d       = int(cfg.get("delay_d", 0))

    gossip_desc = f"schedule {gossip_sched}" if gossip_sched else f"{gossip_steps} gossip steps/round"

    print("\n" + "="*70)
    print(f"EXPERIMENT: {exp_name}")
    print("="*70)
    print(f"  clients={n_clients}, rounds={rounds}, epochs={epochs}, {gossip_desc}")
    print(f"  delay_d={delay_d}")
    print(f"  dataset={dataset}, model={model}")

    topology_path  = resolve_topology(cfg)
    init_weights   = resolve_init_weights(cfg)
    partition_path = resolve_partition(cfg)

    # Which methods to run
    requested = cfg.get("methods")
    if requested:
        methods = [(m, d) for m, d in ALL_METHODS if m in requested]
        missing = set(requested) - {m for m, _ in ALL_METHODS}
        if missing:
            print(f"  WARNING: unknown method(s) in YAML config: {missing}")
    else:
        methods = ALL_METHODS

    print(f"  methods  : {[m for m, _ in methods]}")
    print("="*70)

    results = []
    exp_start = datetime.now()
    total = len(methods)

    for i, (method, description) in enumerate(methods, 1):
        print(f"\n  [{i}/{total}] {description}")
        full_exp_name = f"{exp_name}__{method}"

        cmd = [
            sys.executable, "main.py",
            "--type",            "decentralized",
            "--mixing_method",   method,
            "--num_clients",     str(n_clients),
            "--rounds",          str(rounds),
            "--epochs",          str(epochs),
            "--delay_d",         str(delay_d),
            "--dataset",         dataset,
            "--model",           model,
            "--experiment_name", full_exp_name,
            "--topology_file",   str(topology_path),
            "--init_weights",    str(init_weights),
            "--partition_file",  str(partition_path),
        ]
        if gossip_sched:
            cmd += ["--gossip_schedule", gossip_sched]
        else:
            cmd += ["--gossip_steps", str(gossip_steps)]

        method_start = datetime.now()
        try:
            subprocess.run(cmd, check=True)
            status = "SUCCESS"
        except subprocess.CalledProcessError as e:
            status = f"FAILED (exit {e.returncode})"
        duration = (datetime.now() - method_start).total_seconds() / 60
        print(f"  → {status} ({duration:.1f} min)")
        results.append((exp_name, method, description, status, full_exp_name))

    exp_duration = (datetime.now() - exp_start).total_seconds() / 60
    print(f"\n  Experiment '{exp_name}' finished in {exp_duration:.1f} min")
    return results


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run mixing-method comparison, optionally driven by a YAML experiments file."
    )
    parser.add_argument(
        "--experiments_yaml", type=str, default=None,
        metavar="FILE",
        help="YAML file with a list of experiment configs (see experiments.yaml for format)."
    )
    args = parser.parse_args()

    all_results = []
    global_start = datetime.now()

    if args.experiments_yaml:
        yaml_path = Path(args.experiments_yaml)
        if not yaml_path.exists():
            sys.exit(f"Error: experiments file not found: {yaml_path}")
        with open(yaml_path) as f:
            doc = yaml.safe_load(f)
        experiments = doc.get("experiments", [])
        if not experiments:
            sys.exit("Error: YAML file has no 'experiments' list.")
        print(f"Loaded {len(experiments)} experiment(s) from {yaml_path}")
        for raw in experiments:
            cfg = {**DEFAULTS, **raw}
            all_results.extend(run_experiment(cfg))
    else:
        # Backward-compatible single run with hardcoded DEFAULTS
        cfg = {**DEFAULTS, "name": "default"}
        all_results.extend(run_experiment(cfg))

    total_time = (datetime.now() - global_start).total_seconds() / 60

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for exp, method, desc, status, full_name in all_results:
        mark = "✓" if status == "SUCCESS" else "✗"
        print(f"  {mark} [{exp}] {desc:40s} {status}")
    print(f"\nTotal wall time: {total_time:.1f} min")
    print("="*70)


if __name__ == "__main__":
    main()
