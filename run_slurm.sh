#!/bin/bash -l
#SBATCH --job-name=fl_hypparams_runner
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@600

# ─── setup ────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to cd to $SLURM_SUBMIT_DIR"; exit 1; }
mkdir -p logs

# Initialize conda (works in non-interactive shells)
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
eval "$(conda shell.bash hook 2>/dev/null)" || true

VENV_DIR="fl_hypparams_runner_env"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    echo "Reusing existing venv: ${VENV_DIR}"
else
    if [ -d "${VENV_DIR}" ]; then
        echo "Existing venv is incomplete/corrupted; recreating: ${VENV_DIR}"
        rm -rf "${VENV_DIR}"
    fi
    echo "Creating virtual environment: ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

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

# ─── run delay-runner experiment ──────────────────────────────
echo ""
echo "=========================================="
echo "Starting hyperparams test Experiment"
echo "=========================================="
echo "This will run:"
echo "  1. Decentralized (P2P) - Requested mixing methods from YAML"
echo "  2. Config loaded from experiments_hypparams_runner.yaml"
echo "=========================================="
echo ""

python3 -u run_all_mixing_methods.py --experiments_yaml experiments_delay_runner.yaml

echo ""
echo "Job finished at $(date)"
