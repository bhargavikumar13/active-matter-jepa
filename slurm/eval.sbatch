#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/logs/eval_%j.out
#SBATCH --error=/scratch/%u/logs/eval_%j.err

bash /scratch/$USER/data/active_matter/eval.sh \
    /scratch/$USER/data/active_matter/checkpoints/jepa/best.pt
