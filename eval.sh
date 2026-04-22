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

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
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
