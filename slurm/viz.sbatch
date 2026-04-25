#!/bin/bash
#SBATCH --job-name=embedding_viz
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/logs/viz_%j.out
#SBATCH --error=/scratch/%u/logs/viz_%j.err

OVERLAY=/scratch/$USER/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

singularity exec --nv --overlay $OVERLAY:ro $IMAGE /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate active_matter
    pip install -q umap-learn
    cd /scratch/$USER/data/active_matter
    mkdir -p figures
    python scripts/visualize_embeddings.py \
        --config configs/jepa_run5.yaml \
        --checkpoint checkpoints/jepa/best.pt \
        --output figures/embedding_viz.png \
        --method umap \
        --split test
"
