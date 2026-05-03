#!/bin/bash
# =============================================================================
# eval.sh — Evaluate a JEPA checkpoint on the active_matter dataset
#
# Usage:
#   bash eval.sh [checkpoint_path]
#
# If no checkpoint is provided, defaults to:
#   checkpoints/jepa/best.pt
#
# IMPORTANT: This script requires a GPU node.
# Do NOT run on the login node — feature extraction will OOM.
# Recommended: sbatch slurm/probe.sbatch
#
# Outputs:
#   Linear probe test MSE (combined, alpha, zeta)
#   kNN regression test MSE (combined, alpha, zeta)
#
# Results are saved to:
#   checkpoints/jepa/probe/probe_results__eval__mean.yaml
#   checkpoints/jepa/probe/knn_results__eval__mean.yaml
# =============================================================================
set -e

CHECKPOINT=${1:-/scratch/$USER/data/active_matter/checkpoints/jepa/best.pt}
CONFIG=/scratch/$USER/data/active_matter/configs/jepa_run5.yaml
OVERLAY=/scratch/$USER/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

echo "=============================================="
echo "  JEPA Evaluation"
echo "  Checkpoint : $CHECKPOINT"
echo "  Config     : $CONFIG"
echo "  Date       : $(date)"
echo "=============================================="

# Check checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Download from Google Drive:"
    echo "  pip install gdown"
    echo "  mkdir -p checkpoints/jepa"
    echo "  python -m gdown https://drive.google.com/uc?id=1OhsF3AxGBAvdDQQ60t8KW1K702V9gF5j -O checkpoints/jepa/best.pt"
    exit 1
fi

# Check overlay exists
if [ ! -f "$OVERLAY" ]; then
    echo "ERROR: Overlay not found at $OVERLAY"
    echo "See ENV.md for setup instructions."
    exit 1
fi

# Check for GPU
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected."
    echo "This script must be run on a GPU node, not the login node."
    echo "Use: sbatch slurm/probe.sbatch"
    exit 1
fi

cd /scratch/$USER/data/active_matter

singularity exec --nv \
    --overlay $OVERLAY:ro \
    $IMAGE /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate active_matter
    cd /scratch/$USER/data/active_matter

    echo ''
    echo '── Linear Probe ──────────────────────────────'
    python scripts/probe.py \
        --config $CONFIG \
        --checkpoint $CHECKPOINT \
        --no-wandb

    echo ''
    echo '── kNN Regression ────────────────────────────'
    python scripts/eval_knn.py \
        --config $CONFIG \
        --checkpoint $CHECKPOINT \
        --no-wandb

    echo ''
    echo '=============================================='
    echo '  Evaluation complete.'
    echo '=============================================='
"
