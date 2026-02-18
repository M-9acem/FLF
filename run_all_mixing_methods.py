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
ROUNDS = 500
EPOCHS = 1
GOSSIP_STEPS = 5  # Number of gossip iterations per round
DATASET = "cifar10"
MAIN_LINK_PROB = 1.0
BORDER_LINK_PROB = 1.0
INTRA_CLUSTER_PROB = 0.8

print("="*70)
print("RUNNING ALL MIXING METHODS SEQUENTIALLY")
print("="*70)
print(f"Configuration: {NUM_CLIENTS} clients, {ROUNDS} rounds, {EPOCHS} epochs, {GOSSIP_STEPS} gossip steps/round")
print(f"Dataset: {DATASET}")
print("="*70)

# Generate the topology ONCE
print("\n" + "="*70)
print("GENERATING SHARED TOPOLOGY (used by all methods)")
print("="*70)
graph = create_two_cluster_topology(
    num_clients=NUM_CLIENTS,
    main_link_prob=MAIN_LINK_PROB,
    border_link_prob=BORDER_LINK_PROB,
    intra_cluster_prob=INTRA_CLUSTER_PROB
)

# Save topology to a file that all experiments will use
shared_topology_path = Path("shared_topology.pkl")
with open(shared_topology_path, 'wb') as f:
    pickle.dump(graph, f)
print(f"\nTopology saved to: {shared_topology_path}")
print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
print("All mixing methods will use this same topology!")
print("="*70)

methods = [
    ("metropolis_hastings", "Metropolis-Hastings (Default)"),
    #("max_degree", "Max-Degree (Uniform Weights)"),
    #("jaccard", "Jaccard Similarity"),
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
        "--model", "resnet18",
        "--experiment_name", experiment_name,
        "--topology_file", str(shared_topology_path),
        "--gossip_steps", str(GOSSIP_STEPS)
    ]
    
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

# Clean up temporary topology file
print("\nCleaning up temporary files...")
if shared_topology_path.exists():
    shared_topology_path.unlink()
    print(f"Deleted: {shared_topology_path}")

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
