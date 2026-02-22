# SLURM Templates for TrackIQ

This directory contains batch-script templates for running MiniCluster on HPC clusters.

## Files

- `submit_minicluster.sh`: runs distributed validation (`minicluster run`)
- `submit_bench_collective.sh`: runs fabric-only collective benchmark (`minicluster bench-collective`)

## Submit

```bash
sbatch slurm/submit_minicluster.sh
sbatch slurm/submit_bench_collective.sh
```

## Change node scale

Edit these headers in the scripts:

- `#SBATCH --nodes=<N>`
- `#SBATCH --ntasks-per-node=<T>`
- `#SBATCH --gres=gpu:8` (adjust if your node has a different GPU count)

Total workers are passed as `--workers $SLURM_NTASKS`.

## Backend note (NCCL / RCCL)

Both scripts use `--backend nccl`.
On AMD ROCm clusters, this nccl alias targets RCCL.

For AMD MI300X clusters, set RCCL_DEBUG=INFO to capture collective communication logs for debugging.

