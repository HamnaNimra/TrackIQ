#!/usr/bin/env bash
#
# SLURM template: MiniCluster collective fabric benchmark

#SBATCH --job-name=minicluster_bench_collective
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --output=logs/bench_collective_%j.out

set -euo pipefail

mkdir -p logs results

# Activate your runtime environment (edit one block as needed).
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "Using active conda env: ${CONDA_DEFAULT_ENV}"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate trackiq || true
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"
export MASTER_PORT=29500

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"

srun python -m minicluster bench-collective \
  --workers "${SLURM_NTASKS}" \
  --backend nccl \
  --size-mb 256 \
  --iterations 50 \
  --output "results/bench_collective_${SLURM_JOB_ID}.json"

