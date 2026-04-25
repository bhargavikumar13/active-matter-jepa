#!/bin/bash
#SBATCH --job-name=jepa_probe
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/logs/probe_%j.out
#SBATCH --error=/scratch/%u/logs/probe_%j.err

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Start time : $(date)"

mkdir -p /scratch/$USER/logs

OVERLAY=/scratch/$USER/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
CKPT=/scratch/$USER/data/active_matter/checkpoints/jepa/best.pt

singularity exec --nv \
    --overlay $OVERLAY:ro \
    $IMAGE /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate active_matter
    cd /scratch/$USER/data/active_matter

    echo '── Linear probe ──'
    python scripts/probe.py \
        --config configs/jepa.yaml \
        --checkpoint $CKPT \
        --no-wandb

    echo '── kNN regression ──'
    python scripts/eval_knn.py \
        --config configs/jepa.yaml \
        --checkpoint $CKPT \
        --k 1 5 10 20 50 \
        --no-wandb
"

echo "End time : $(date)"
