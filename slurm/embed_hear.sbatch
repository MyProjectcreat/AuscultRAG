#!/usr/bin/env bash
# HeAR embeddings for all units on one GPU (resumable: resubmit to continue). Submit from the repo root;
# set the partition / GPU type for your cluster.
#SBATCH --job-name=ar_hear
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=embed_hear_%j.log
cd "${SLURM_SUBMIT_DIR:-.}"
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python scripts/embed_units_hear.py --batch 128 --workers 10
