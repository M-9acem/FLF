#!/bin/bash -l
#SBATCH --job-name=fl_hparam
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@600

eval "$(conda shell.bash hook 2>/dev/null)" || true
set -euo pipefail

# This script has two modes:
# 1) Launcher mode (no SLURM_JOB_ID): submit one job per YAML experiment file.
# 2) Worker mode   (with SLURM_JOB_ID): run one experiment YAML in this allocation.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

VENV_PATH="${VENV_PATH:-$ROOT_DIR/venv}"
YAML_GLOB="${YAML_GLOB:-config/hparam_sweeps/single/*.yaml}"

resolve_path() {
    local candidate="$1"
    if [[ -f "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/$candidate" ]]; then
        printf '%s\n' "$SLURM_SUBMIT_DIR/$candidate"
        return 0
    fi
    if [[ -f "$ROOT_DIR/$candidate" ]]; then
        printf '%s\n' "$ROOT_DIR/$candidate"
        return 0
    fi
    return 1
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Launcher mode: submitting one job per experiment YAML"
    shopt -s nullglob
    yaml_files=( $YAML_GLOB )
    shopt -u nullglob

    if [[ ${#yaml_files[@]} -eq 0 ]]; then
        echo "No YAML files matched: $YAML_GLOB"
        exit 1
    fi

    for yaml in "${yaml_files[@]}"; do
        exp_name="$(basename "$yaml" .yaml)"
        abs_yaml="$(resolve_path "$yaml")"
        if [[ -z "$abs_yaml" ]]; then
            echo "Could not resolve YAML path: $yaml"
            exit 1
        fi
        echo "Submitting: $yaml"
        sbatch \
            --job-name="fl_${exp_name}" \
            --export=ALL,EXP_YAML="$abs_yaml",VENV_PATH="$VENV_PATH" \
            "$0"
    done

    echo "Submitted ${#yaml_files[@]} jobs."
    exit 0
fi

# Worker mode
if [[ -z "${EXP_YAML:-}" ]]; then
    echo "EXP_YAML is not set."
    echo "Submit with: sbatch --export=ALL,EXP_YAML=config/hparam_sweeps/single/<file>.yaml run_slurm.sh"
    exit 1
fi

# Determine the project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
cd "$ROOT_DIR"

EXP_YAML="$(resolve_path "$EXP_YAML" || true)"
if [[ -z "$EXP_YAML" ]]; then
    echo "YAML file not found: ${EXP_YAML:-<unset>}"
    exit 1
fi

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    echo "Python venv not found at: $VENV_PATH"
    echo "Set VENV_PATH=/path/to/venv when launching."
    exit 1
fi

source "$VENV_PATH/bin/activate"

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Experiment YAML: $EXP_YAML"

echo ""
echo "Installing/updating requirements..."
python -m pip install -r "$ROOT_DIR/requirements.txt"

echo ""
nvidia-smi || true

python -u -c "
import torch
print('torch        ', torch.__version__)
print('cuda avail   ', torch.cuda.is_available())
print('cuda devices ', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu name     ', torch.cuda.get_device_name(0))
"

echo ""
echo "Running experiment from: $EXP_YAML"
python -u run_all_mixing_methods.py --experiments_yaml "$EXP_YAML"

echo ""
echo "Job finished at $(date)"
