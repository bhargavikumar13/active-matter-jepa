#!/bin/bash
# run.sh — Enter the Singularity container with the active_matter conda env
#
# Usage:
#   ./run.sh                        # interactive shell
#   ./run.sh python scripts/train.py --config configs/jepa.yaml
#   ./run.sh python src/model.py

OVERLAY=/scratch/$USER/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

# If no arguments given, drop into an interactive bash shell
if [ $# -eq 0 ]; then
    singularity exec --nv \
        --overlay $OVERLAY:ro \
        $IMAGE \
        /bin/bash -c "source /ext3/env.sh && conda activate active_matter && exec bash"
else
    # Run the command passed as arguments
    singularity exec --nv \
        --overlay $OVERLAY:ro \
        $IMAGE \
        /bin/bash -c "source /ext3/env.sh && conda activate active_matter && $*"
fi
