#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")" || exit 1

YAML_DIR="exp_yamls/lr_0_1"
RUN_SCRIPT="run_slurm.sh"

if [ ! -d "${YAML_DIR}" ]; then
    echo "ERROR: Directory ${YAML_DIR} not found" >&2
    exit 1
fi

if [ ! -f "${RUN_SCRIPT}" ]; then
    echo "ERROR: ${RUN_SCRIPT} not found" >&2
    exit 1
fi

mapfile -t YAML_FILES < <(find "${YAML_DIR}" -maxdepth 1 -type f -name "*.yaml" | sort)
TOTAL_FILES=${#YAML_FILES[@]}

if [ "${TOTAL_FILES}" -eq 0 ]; then
    echo "ERROR: No YAML files found in ${YAML_DIR}" >&2
    exit 1
fi

echo "Submitting ${TOTAL_FILES} separate jobs from ${YAML_DIR}"

for idx in "${!YAML_FILES[@]}"; do
    YAML_FILE="${YAML_FILES[$idx]}"
    EXPERIMENT_NUM=$((idx + 1))
    YAML_BASENAME="$(basename "${YAML_FILE}" .yaml)"
    JOB_NAME="lr01_${YAML_BASENAME}"
    # Keep a safe size for schedulers that limit job-name length.
    JOB_NAME="${JOB_NAME:0:120}"

    echo "[${EXPERIMENT_NUM}/${TOTAL_FILES}] Submitting ${YAML_FILE} as job ${JOB_NAME}"
    sbatch --job-name="${JOB_NAME}" --export=ALL,EXPERIMENTS_YAML="${YAML_FILE}" "${RUN_SCRIPT}"
done

echo "All jobs submitted."
