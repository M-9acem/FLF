"""
Run full comparison: Decentralized (all mixing methods) vs Centralized
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"STARTING: {description}")
    print("="*80)
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            shell=True,
            text=True
        )
        print(f"\n✓ COMPLETED: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ FAILED: {description}")
        print(f"Error: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("FULL FEDERATED LEARNING COMPARISON")
    print("="*80)
    print("\nThis will run:")
    print("  1. Decentralized (P2P) with all mixing methods")
    print("  2. Centralized (FedAvg)")
    print("\nConfiguration:")
    print("  - Clients: 40")
    print("  - Rounds: 600")
    print("  - Epochs: 6")
    print("="*80)
    
    # Step 1: Run decentralized experiments with all mixing methods
    decentralized_success = run_command(
        "python run_all_mixing_methods.py",
        "Decentralized (P2P) - All Mixing Methods"
    )
    
    if not decentralized_success:
        print("\n⚠ Warning: Decentralized experiments failed, but continuing with centralized...")
    
    # Step 2: Run centralized experiment
    centralized_cmd = (
        "python main.py "
        "--type centralized "
        "--num_clients 40 "
        "--rounds 600 "
        "--epochs 6"
    )
    
    centralized_success = run_command(
        centralized_cmd,
        "Centralized (FedAvg) - 40 clients, 600 rounds, 6 epochs"
    )
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Decentralized (P2P):  {'✓ SUCCESS' if decentralized_success else '✗ FAILED'}")
    print(f"Centralized (FedAvg): {'✓ SUCCESS' if centralized_success else '✗ FAILED'}")
    print("="*80)
    
    if decentralized_success and centralized_success:
        print("\n🎉 All experiments completed successfully!")
        print("\nResults are available in:")
        print("  - logs/matcha_comparison/ (decentralized with all mixing methods)")
        print("  - logs/centralized_fedavg/ (centralized)")
        print("\nAnalysis notebooks:")
        print("  - analysis_p2p_simple.ipynb (for decentralized)")
        print("  - analysis_centralized_simple.ipynb (for centralized)")
        return 0
    else:
        print("\n⚠ Some experiments failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
