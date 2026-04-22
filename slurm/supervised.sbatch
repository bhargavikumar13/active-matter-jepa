#!/bin/bash
#SBATCH --job-name=supervised
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=47:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/logs/supervised_%j.out
#SBATCH --error=/scratch/%u/logs/supervised_%j.err

# ── Preemption Handler ────────────────────────────────────────────────────────
cleanup_and_requeue() {
    echo "Preemption detected at $(date). Requeueing job $SLURM_JOB_ID..."
    scontrol requeue $SLURM_JOB_ID
    kill $SINGULARITY_PID 2>/dev/null
    exit 143
}
trap 'cleanup_and_requeue' SIGTERM SIGINT

# ── Environment ──────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Start time : $(date)"

mkdir -p /scratch/$USER/logs
mkdir -p /scratch/$USER/data/active_matter/checkpoints/supervised

OVERLAY=/scratch/$USER/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

singularity exec --nv \
    --overlay $OVERLAY:ro \
    $IMAGE /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate active_matter

    cd /scratch/$USER/data/active_matter

    export WANDB_API_KEY=YOUR_WANDB_API_KEY

    echo 'Starting supervised baseline...'
    python scripts/supervised.py \
        --config configs/jepa.yaml
" &
SINGULARITY_PID=$!
wait $SINGULARITY_PID

echo "End time : $(date)"
