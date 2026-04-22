#!/bin/bash
#SBATCH --job-name=competition
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/logs/competition_%j.out
#SBATCH --error=/scratch/%u/logs/competition_%j.err

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Start time : $(date)"

OVERLAY=/scratch/$USER/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
BASE=/scratch/$USER/data/active_matter
CKPT=$BASE/checkpoints/jepa/best.pt
CFG=$BASE/configs/jepa_run5.yaml

singularity exec --nv --overlay $OVERLAY:ro $IMAGE /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate active_matter
    cd $BASE

    echo '======================================================='
    echo '1. Separate probes for alpha and zeta'
    echo '======================================================='
    python scripts/probe_separate.py \
        --config $CFG \
        --checkpoint $CKPT \
        --no-wandb

    echo '======================================================='
    echo '2. L2 regularization sweep'
    echo '======================================================='
    python scripts/probe_sweep.py \
        --config $CFG \
        --checkpoint $CKPT \
        --weight_decays 1e-5 1e-4 1e-3 1e-2 1e-1 \
        --no-wandb

    echo '======================================================='
    echo '3. 5-fold CV with ensemble test predictions'
    echo '======================================================='
    python scripts/probe_cv.py \
        --config $CFG \
        --checkpoint $CKPT \
        --n_folds 5 \
        --ensemble_test \
        --no-wandb

    echo '======================================================='
    echo '4. Epoch ensemble (epoch 80 + epoch 90 + best)'
    echo '======================================================='
    python scripts/probe_ensemble_checkpoints.py \
        --config $CFG \
        --checkpoints \
            $BASE/checkpoints/jepa/epoch_0080.pt \
            $BASE/checkpoints/jepa/epoch_0090.pt \
            $BASE/checkpoints/jepa/best.pt \
        --no-wandb

    echo '======================================================='
    echo '5. Attention pooling probe'
    echo '======================================================='
    python scripts/attention_pool_probe.py \
        --config $CFG \
        --checkpoint $CKPT \
        --no-wandb

    echo '======================================================='
    echo 'All competition scripts complete'
    echo '======================================================='
"

echo "End time : $(date)"
