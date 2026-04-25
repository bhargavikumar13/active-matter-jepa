#!/bin/bash
# monitor.sh — Auto-monitor and resubmit training job if preempted
#
# Usage:
#   chmod +x monitor.sh
#   sbatch --account=csci_ga_2572-2026sp \
#       --partition=n2c48m24 \
#       --time=47:00:00 \
#       --output=/scratch/$USER/logs/monitor_%j.out \
#       --wrap="cd /scratch/$USER/data/active_matter && bash monitor.sh"
#
# To stop:
#   scancel <monitor_job_id>

SBATCH_SCRIPT="/scratch/$USER/data/active_matter/slurm/train.sbatch"
CHECKPOINT="/scratch/$USER/data/active_matter/checkpoints/jepa/latest.pt"
LOG_FILE="/scratch/$USER/logs/monitor.log"
PID_FILE="/scratch/$USER/logs/monitor.pid"
CHECK_INTERVAL=30
USER="$USER"

# Save PID
echo $$ > $PID_FILE
echo "Monitor started (PID=$$)" | tee -a $LOG_FILE

get_job_status() {
    # Returns the job name if ANY jepa job exists in squeue, regardless of status
    squeue -u $USER -h -o "%j" 2>/dev/null | grep "jepa" | head -1
}

get_epoch() {
    if [ -f "$CHECKPOINT" ]; then
        python3 -c "
import torch
try:
    ckpt = torch.load('$CHECKPOINT', map_location='cpu')
    print(ckpt.get('epoch', -1))
except:
    print(-1)
" 2>/dev/null
    else
        echo "-1"
    fi
}

while true; do
    STATUS=$(get_job_status)
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    if [ -n "$STATUS" ]; then
        # Any jepa job exists in queue — do nothing
        echo "[$TIMESTAMP] Job exists in queue — waiting..." | tee -a $LOG_FILE
    else
        # No jepa job found at all — check if training is complete
        EPOCH=$(get_epoch)
        echo "[$TIMESTAMP] No jepa job found. Last epoch: $EPOCH" | tee -a $LOG_FILE

        if [ "$EPOCH" -ge 99 ] 2>/dev/null; then
            echo "[$TIMESTAMP] Training complete at epoch $EPOCH. Stopping monitor." | tee -a $LOG_FILE
            rm -f $PID_FILE
            exit 0
        else
            echo "[$TIMESTAMP] Training incomplete (epoch $EPOCH/49). Resubmitting..." | tee -a $LOG_FILE
            NEW_JOB=$(sbatch $SBATCH_SCRIPT 2>&1)
            echo "[$TIMESTAMP] $NEW_JOB" | tee -a $LOG_FILE
        fi
    fi

    sleep $CHECK_INTERVAL
done
