#!/usr/bin/env python3
"""Compare SSOS vs Standard Gossip test results."""

import pandas as pd
from pathlib import Path
import json

# Load results from previous tests
standard_gossip_dir = Path("logs/comparison_standard_gossip_gs3_r5_n4__max_degree/2026-05-12_11-33-25")
ssos_gossip_dir = Path("logs/smoke_test_ssos_gs3_r5_n4__max_degree/2026-05-12_11-16-59")

print("="*80)
print("COMPARISON: SSOS vs Standard Gossip")
print("="*80)
print()

# Load global aggregated metrics
if standard_gossip_dir.exists():
    std_metrics = pd.read_csv(standard_gossip_dir / "global_aggregated_metrics.csv")
    print("STANDARD GOSSIP (3 steps per round):")
    print(std_metrics[['round', 'gossip_step', 'test_accuracy', 'test_loss']])
else:
    print("Standard gossip results not found in:", standard_gossip_dir)

print()
print("-"*80)
print()

if ssos_gossip_dir.exists():
    ssos_metrics = pd.read_csv(ssos_gossip_dir / "global_aggregated_metrics.csv")
    print("SSOS GOSSIP (3 steps per round with momentum, g=0.171573):")
    print(ssos_metrics[['round', 'gossip_step', 'test_accuracy', 'test_loss']])
else:
    print("SSOS gossip results not found in:", ssos_gossip_dir)

print()
print("="*80)
print("COMPARISON SUMMARY")
print("="*80)

if standard_gossip_dir.exists() and ssos_gossip_dir.exists():
    # Get final accuracies per round
    std_final = std_metrics[std_metrics['gossip_step'] == std_metrics.groupby('round')['gossip_step'].transform('max')].reset_index(drop=True)
    ssos_final = ssos_metrics[ssos_metrics['gossip_step'] == ssos_metrics.groupby('round')['gossip_step'].transform('max')].reset_index(drop=True)
    
    print()
    print("Final Round Accuracies (after all gossip steps):")
    print("-"*80)
    print(f"{'Round':<8} {'Standard':<12} {'SSOS':<12} {'Delta':<12} {'SSOS Better?':<15}")
    print("-"*80)
    
    for round_num in std_final['round'].unique():
        std_acc = std_final[std_final['round'] == round_num]['test_accuracy'].values[0]
        ssos_acc = ssos_final[ssos_final['round'] == round_num]['test_accuracy'].values[0]
        delta = ssos_acc - std_acc
        better = "✓ YES" if delta > 0 else "✗ NO"
        print(f"{round_num:<8} {std_acc:<12.2f} {ssos_acc:<12.2f} {delta:+<12.2f} {better:<15}")
    
    print("-"*80)
    
    # Overall comparison
    std_avg = std_final['test_accuracy'].mean()
    ssos_avg = ssos_final['test_accuracy'].mean()
    overall_delta = ssos_avg - std_avg
    
    print(f"{'AVERAGE':<8} {std_avg:<12.2f} {ssos_avg:<12.2f} {overall_delta:+<12.2f}")
    print()
    
    print("="*80)
    print("DEBUG: Monitor Within-Round Gossip Step Progression")
    print("="*80)
    print()
    print("STANDARD GOSSIP - Gossip step progression (no momentum):")
    print("-"*80)
    for round_num in sorted(std_metrics['round'].unique()):
        round_data = std_metrics[std_metrics['round'] == round_num].sort_values('gossip_step')
        accs = round_data['test_accuracy'].values
        print(f"Round {round_num}: Steps [", end="")
        print(" → ".join([f"{acc:.1f}%" for acc in accs]), end="")
        print("]")
    
    print()
    print("SSOS GOSSIP - Gossip step progression (with momentum):")
    print("-"*80)
    for round_num in sorted(ssos_metrics['round'].unique()):
        round_data = ssos_metrics[ssos_metrics['round'] == round_num].sort_values('gossip_step')
        accs = round_data['test_accuracy'].values
        print(f"Round {round_num}: Steps [", end="")
        print(" → ".join([f"{acc:.1f}%" for acc in accs]), end="")
        print("]")
    
    print()
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    if overall_delta > 0.5:
        print(f"✓ SSOS is BETTER: +{overall_delta:.2f}% average accuracy improvement")
        print("  Momentum-accelerated gossip is helping convergence.")
    elif overall_delta > -0.5:
        print(f"~ SSOS is COMPARABLE: {overall_delta:+.2f}% difference (within noise margin)")
        print("  Momentum effect is neutral or masked by stochasticity.")
    else:
        print(f"✗ SSOS is WORSE: {overall_delta:.2f}% accuracy degradation")
        print("  Momentum may be overshooting or causing divergence.")
    
    print()
    print("NOTE: SSOS is designed to accelerate CONSENSUS over many gossip steps.")
    print("      With only 3 steps per round, the effect may be subtle.")
    print("      A test with more gossip steps (e.g., 9+) would show clearer benefits.")
