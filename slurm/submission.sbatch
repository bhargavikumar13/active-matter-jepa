#!/bin/bash
#SBATCH --job-name=submission
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/logs/submission_%j.out
#SBATCH --error=/scratch/%u/logs/submission_%j.err

OVERLAY=/scratch/$USER/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

singularity exec --nv --overlay $OVERLAY:ro $IMAGE /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate active_matter
    cd /scratch/$USER/data/active_matter

    echo '── Val check ──'
    python scripts/generate_submission.py \
        --config configs/jepa_run5.yaml \
        --checkpoint checkpoints/jepa/best.pt \
        --probe checkpoints/jepa/probe/linear_probe__best__mean.pt \
        --split val \
        --output submissions/val_check.csv

    echo '── Ensemble + TTA + calibration ──'
    python scripts/generate_submission.py \
        --config configs/jepa_run5.yaml \
        --checkpoint checkpoints/jepa/best.pt \
                      checkpoints/jepa_run4/best.pt \
                      checkpoints/jepa_run3/best.pt \
        --probe checkpoints/jepa/probe/linear_probe__best__mean.pt \
                checkpoints/jepa_run4/probe/linear_probe__best__mean.pt \
                checkpoints/jepa_run3/probe/linear_probe__best__mean.pt \
        --config2 configs/jepa_run4.yaml configs/jepa_run3.yaml \
        --tta 5 \
        --calibrate \
        --output submissions/submission_ensemble_tta_cal.csv
"
