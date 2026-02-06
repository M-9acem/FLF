#!/bin/bash -l
#SBATCH --job-name=fl_full_comparison
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=B:TERM@600

# ─── setup ────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to cd to $SLURM_SUBMIT_DIR"; exit 1; }
mkdir -p logs

# Initialize conda (works in non-interactive shells)
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
eval "$(conda shell.bash hook 2>/dev/null)" || true

conda activate my_fatfl_env || { echo "Failed to activate conda env"; exit 1; }

# ─── diagnostics ──────────────────────────────────────────────
echo "Job started on $(hostname) at $(date)"
nvidia-smi || true
python -u -c "
import torch
print('torch        ', torch.__version__)
print('cuda avail   ', torch.cuda.is_available())
print('cuda devices ', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu name     ', torch.cuda.get_device_name(0))
"

# ─── install requirements ─────────────────────────────────────
echo ""
echo "Installing/updating requirements..."
pip install -r requirements.txt || echo "Warning: requirements.txt not found or install failed"

# ─── run full comparison ──────────────────────────────────────
echo ""
echo "=========================================="
echo "Starting Full Comparison Experiment"
echo "=========================================="
echo "This will run:"
echo "  1. Decentralized (P2P) - All mixing methods"
echo "  2. Centralized (FedAvg) - 40 clients, 400 rounds, 4 epochs"
echo "=========================================="
echo ""

python3 -u run_full_comparison.py

conda deactivate
echo ""
echo "Job finished at $(date)"
