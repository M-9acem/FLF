#!/bin/bash -l
#SBATCH --job-name=SSOS_MULTI_GOSSIP
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=120:30:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
JOB_TAG="${SLURM_JOB_ID:-local}_${RUN_TS}"
DEBUG_DIR="logs/slurm_debug/${JOB_TAG}"
VENV_DIR="fl_hypparams_runner_env"

mkdir -p "${DEBUG_DIR}"

exec > >(tee -a "${DEBUG_DIR}/slurm_run.log") 2>&1

cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "Failed to cd to submit directory"; exit 1; }
mkdir -p logs

echo "Debug directory: ${DEBUG_DIR}"
echo "Job started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "SLURM job id: ${SLURM_JOB_ID:-local}"

echo
echo "===== SLURM / CUDA ENV ====="
env | grep -E '^(SLURM|CUDA|NVIDIA|CONDA|VIRTUAL_ENV)_' | sort || true

echo
echo "===== SYSTEM GPU DIAGNOSTICS ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi || true
else
  echo "nvidia-smi not found in PATH"
fi

echo
echo "===== PYTHON SETUP ====="

# Try to load a newer Python via module system (common on HPC)
echo "Attempting to load Python via module system..."
if command -v module >/dev/null 2>&1; then
  for py_module in python/3.10 python/3.9 python/3.8 python-3.10 python-3.9 python-3.8; do
    if module load "${py_module}" 2>/dev/null; then
      echo "  ✓ Loaded module: ${py_module}"
      break
    fi
  done
fi

PYTHON_BIN=""
for candidate in python3.11 python3.10 python3.9 python3; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "${candidate}")"
    break
  fi
done

if [ -z "${PYTHON_BIN}" ]; then
  echo "No usable Python interpreter found in PATH" >&2
  exit 1
fi

# Verify Python version is 3.8+
PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')"
PYTHON_MAJOR="$(${PYTHON_BIN} -c 'import sys; print(sys.version_info[0])')"
PYTHON_MINOR="$(${PYTHON_BIN} -c 'import sys; print(sys.version_info[1])')"

if [ "${PYTHON_MAJOR}" -lt 3 ] || { [ "${PYTHON_MAJOR}" -eq 3 ] && [ "${PYTHON_MINOR}" -lt 8 ]; }; then
  echo "ERROR: Found Python ${PYTHON_VERSION} but torch>=2.0.0 requires Python 3.8+" >&2
  echo "Please load a newer Python via: module load python/3.9 (or similar)" >&2
  exit 1
fi

echo "Selected base Python: ${PYTHON_BIN} (${PYTHON_VERSION})"

if [ -d "${VENV_DIR}" ]; then
  echo "Reusing existing venv: ${VENV_DIR}"
else
  echo "Creating virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo
echo "===== PYTHON RUNTIME DIAGNOSTICS ====="
which python || true
python --version || true
pip --version || true

echo
echo "Installing/updating requirements..."
pip install -r requirements.txt 2>&1 | tee -a "${DEBUG_DIR}/pip_install.log"

echo
echo "===== TORCH DIAGNOSTICS ====="
python -u -c "
import os
import platform
from datetime import datetime

import torch

print('timestamp     ', datetime.now().isoformat())
print('platform      ', platform.platform())
print('python exec   ', os.sys.executable)
print('torch        ', torch.__version__)
print('torch file   ', torch.__file__)
print('cuda avail   ', torch.cuda.is_available())
print('cuda version ', torch.version.cuda)
print('cuda devices ', torch.cuda.device_count())
if torch.cuda.is_available():
  print('gpu name     ', torch.cuda.get_device_name(0))
  print('current dev  ', torch.cuda.current_device())
else:
  print('cuda reason  ', 'CUDA not available from this process')
" | tee "${DEBUG_DIR}/torch_cuda_check.txt"

echo

echo "===== CUDA STRICT CHECK ====="
python -u - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
  print("ERROR: CUDA is not available. Failing fast to avoid CPU training.", file=sys.stderr)
  sys.exit(1)

print(f"CUDA check passed: {torch.cuda.device_count()} GPU(s) visible")
PY

# Start lightweight GPU monitoring so we can prove utilization during training.
GPU_MONITOR_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "===== START GPU MONITOR (5s sampling) ====="
  echo "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw" > "${DEBUG_DIR}/gpu_usage.csv"
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader,nounits \
    -l 5 >> "${DEBUG_DIR}/gpu_usage.csv" &
  GPU_MONITOR_PID=$!
  echo "GPU monitor PID: ${GPU_MONITOR_PID}"
fi

cleanup() {
  if [ -n "${GPU_MONITOR_PID:-}" ] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
    kill "${GPU_MONITOR_PID}" || true
  fi
}
trap cleanup EXIT

echo
echo "===== SMALL SLURM SMOKE EXPERIMENT ====="
EXPERIMENTS_YAML_PATH="exp_yamls/cifar10_resnet8_lr0001_gs9_ssos.yaml"
echo "Using experiments YAML: ${EXPERIMENTS_YAML_PATH}"
python -u run_all_mixing_methods.py --experiments_yaml "${EXPERIMENTS_YAML_PATH}"

echo
echo "Job finished at $(date)"
echo "Debug artifacts saved under: ${DEBUG_DIR}"
