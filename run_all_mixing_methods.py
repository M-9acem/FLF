#!/usr/bin/env python
"""Run all mixing methods sequentially for comparison."""

import subprocess
import sys
from datetime import datetime

# Configuration
NUM_CLIENTS = 10
ROUNDS = 15
EPOCHS = 5
DATASET = "cifar10"

print("="*70)
print("RUNNING ALL MIXING METHODS SEQUENTIALLY")
print("="*70)
print(f"Configuration: {NUM_CLIENTS} clients, {ROUNDS} rounds, {EPOCHS} epochs")
print(f"Dataset: {DATASET}")
print("="*70)

methods = [
    ("metropolis_hastings", "Metropolis-Hastings (Default)"),
    ("max_degree", "Max-Degree (Uniform Weights)"),
    ("jaccard", "Jaccard Similarity"),
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
        "--experiment_name", experiment_name
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

# Print summary
print("\n" + "="*70)
print("EXECUTION SUMMARY")
print("="*70)

for method, description, status, exp_name in results:
    print(f"{description:30s} {status:20s} logs/{exp_name}/")

print(f"\nTotal time: {total_duration:.1f} minutes")
print("="*70)

# Print analysis instructions
print("\n" + "="*70)
print("NEXT STEPS - ANALYZE RESULTS")
print("="*70)
print("\n1. Compare results in logs/ directory:")
for method, _, _, exp_name in results:
    print(f"   - logs/{exp_name}/")

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
