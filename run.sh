#!/bin/bash
# run.sh — Enter the Singularity container with the active_matter conda env
#
# Usage:
#   ./run.sh                        # interactive shell
#   ./run.sh python scripts/train.py --config configs/jepa.yaml
#   ./run.sh python scripts/probe.py --config configs/jepa.yaml --checkpoint checkpoints/jepa/best.pt

OVERLAY=/scratch/$USER/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

# Check overlay exists
if [ ! -f "$OVERLAY" ]; then
    echo "ERROR: Overlay not found at $OVERLAY"
    echo "See ENV.md for setup instructions."
    exit 1
fi

# Write a temporary wrapper script to preserve argument quoting
WRAPPER=$(mktemp /tmp/run_wrapper_XXXXXX.sh)
cat > "$WRAPPER" << WRAPPER_EOF
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate active_matter
WRAPPER_EOF

if [ $# -eq 0 ]; then
    echo "exec bash" >> "$WRAPPER"
else
    # Write each argument safely using printf
    printf 'exec' >> "$WRAPPER"
    for arg in "$@"; do
        printf ' %q' "$arg" >> "$WRAPPER"
    done
    echo >> "$WRAPPER"
fi

chmod +x "$WRAPPER"

singularity exec --nv \
    --overlay $OVERLAY:ro \
    $IMAGE \
    /bin/bash "$WRAPPER"

rm -f "$WRAPPER"
