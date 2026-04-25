#!/bin/bash
#SBATCH --job-name=knn_run3
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/logs/knn_run3_%j.out
#SBATCH --error=/scratch/%u/logs/knn_run3_%j.err

OVERLAY=/scratch/$USER/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

singularity exec --nv --overlay $OVERLAY:ro $IMAGE /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate active_matter
    cd /scratch/$USER/data/active_matter

    python scripts/eval_knn.py \
        --config configs/jepa_run3.yaml \
        --checkpoint checkpoints/jepa_run3/best.pt \
        --k 1 5 10 20 50 \
        --no-wandb
"
