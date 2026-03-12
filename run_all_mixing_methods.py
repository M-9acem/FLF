#!/usr/bin/env python
"""Run all mixing methods sequentially for comparison using the same topology."""

import subprocess
import sys
from datetime import datetime
import pickle
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from src.decentralized.topology import create_two_cluster_topology

# Configuration - QUICK TEST
NUM_CLIENTS = 40
ROUNDS = 200
EPOCHS = 2
GOSSIP_STEPS = 1       # Used only when GOSSIP_SCHEDULE is None
GOSSIP_SCHEDULE = None # e.g. "5:0,3:100,1:200" — overrides GOSSIP_STEPS when set
DATASET = "cifar10"
MAIN_LINK_PROB = 1.0
BORDER_LINK_PROB = 1.0
INTRA_CLUSTER_PROB = 0.8

print("="*70)
print("RUNNING ALL MIXING METHODS SEQUENTIALLY")
print("="*70)
gossip_desc = f"schedule {GOSSIP_SCHEDULE}" if GOSSIP_SCHEDULE else f"{GOSSIP_STEPS} gossip steps/round"
print(f"Configuration: {NUM_CLIENTS} clients, {ROUNDS} rounds, {EPOCHS} epochs, {gossip_desc}")
print(f"Dataset: {DATASET}")
print("="*70)

# Canonical topology — load if exists, generate and save if not
print("\n" + "="*70)
print("SHARED TOPOLOGY (used by all methods)")
print("="*70)
shared_topology_path = Path("shared_topology.pkl")
if shared_topology_path.exists():
    print(f"Loading existing topology from: {shared_topology_path}")
    with open(shared_topology_path, 'rb') as f:
        graph = pickle.load(f)
else:
    print("Topology file not found — generating and saving ...")
    graph = create_two_cluster_topology(
        num_clients=NUM_CLIENTS,
        main_link_prob=MAIN_LINK_PROB,
        border_link_prob=BORDER_LINK_PROB,
        intra_cluster_prob=INTRA_CLUSTER_PROB,
        intra_cluster_communication=False
    )
    with open(shared_topology_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Topology saved to: {shared_topology_path} (will be reused in future runs)")
print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
print("All mixing methods will use this same topology!")
print("="*70)

# Canonical initial weights — generated once by generate_init_weights.py
shared_w0_path = Path('init_weights') / 'resnet8_w0.pt'
if not shared_w0_path.exists():
    print(f'\nInitial weights not found: {shared_w0_path}')
    print('Running generate_init_weights.py to create them ...')
    subprocess.run([sys.executable, 'generate_init_weights.py'], check=True)
print(f'\nUsing canonical initial weights: {shared_w0_path}')
print('All mixing methods will start from these same weights!')
print('='*70)

# Canonical partition — generated once by generate_partition.py
partition_file = Path('data_partition') / f'{DATASET}_N{NUM_CLIENTS}_dirichlet_a0.5.pkl'
if not partition_file.exists():
    print(f'\nPartition file not found: {partition_file}')
    print('Running generate_partition.py to create it ...')
    import subprocess as _sp
    _sp.run([sys.executable, 'generate_partition.py'], check=True)
    if not partition_file.exists():
        raise FileNotFoundError(
            f'generate_partition.py ran but {partition_file} was not created.\n'
            f'Check that CONFIGS in generate_partition.py includes N={NUM_CLIENTS}.'
        )
print(f'\nUsing canonical data partition: {partition_file}')
print('All mixing methods will train on identical client data splits!')
print('='*70)

methods = [
    ("metropolis_hastings", "Metropolis-Hastings (Default)"),
    ("max_degree", "Max-Degree (Uniform Weights)"),
    ("jaccard", "Jaccard Similarity"),
    ("jaccard_dissimilarity", "Jaccard Dissimilarity (Avrachenkov)"),
    ("matcha", "MATCHA (Optimal)")
]

results = []
start_time = datetime.now()

for i, (method, description) in enumerate(methods, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/4] Running: {description}")
    print(f"{'='*70}\n")
    
    experiment_name = f"{method}_comparison"
    
    cmd = [
        sys.executable, "main.py",
        "--type", "decentralized",
        "--mixing_method", method,
        "--num_clients", str(NUM_CLIENTS),
        "--rounds", str(ROUNDS),
        "--epochs", str(EPOCHS),
        "--dataset", DATASET,
        "--model", "resnet8",
        "--experiment_name", experiment_name,
        "--topology_file", str(shared_topology_path),
        "--init_weights", str(shared_w0_path),
        "--partition_file", str(partition_file)
    ]
    if GOSSIP_SCHEDULE:
        cmd += ["--gossip_schedule", GOSSIP_SCHEDULE]
    else:
        cmd += ["--gossip_steps", str(GOSSIP_STEPS)]
    
    method_start = datetime.now()
    
    try:
        result = subprocess.run(cmd, check=True)
        status = "✓ SUCCESS"
        results.append((method, description, status, experiment_name))
    except subprocess.CalledProcessError as e:
        status = f"✗ FAILED (exit code {e.returncode})"
        results.append((method, description, status, experiment_name))
    
    method_end = datetime.now()
    duration = (method_end - method_start).total_seconds() / 60
    print(f"\n{description} completed in {duration:.1f} minutes")

end_time = datetime.now()
total_duration = (end_time - start_time).total_seconds() / 60

# Print summary
print("\n" + "="*70)
print("EXECUTION SUMMARY")
print("="*70)

for method, description, status, exp_name in results:
    print(f"{description:30s} {status:20s} logs_test/{exp_name}/")

print(f"\nTotal time: {total_duration:.1f} minutes")
print("="*70)

# Print analysis instructions
print("\n" + "="*70)
print("NEXT STEPS - ANALYZE RESULTS")
print("="*70)
print("\n1. Compare results in logs/ directory:")
for method, _, _, exp_name in results:
    print(f"   - logs_test/{exp_name}/")

print("\n2. Use analysis.ipynb to visualize each experiment")
print("\n3. Compare convergence rates across methods:")
print("   - Load round_summary.csv from each experiment")
print("   - Plot avg_accuracy vs round for all methods")
print("\n4. Expected differences:")
print("   - Metropolis-Hastings: Baseline, proven convergence")
print("   - Max-Degree: Similar to MH, slightly different weights")
print("   - Jaccard: May converge faster on clustered topology")
print("   - MATCHA: Optimal (if cvxpy available), else same as MH")

print("\n" + "="*70)
