"""
scripts/compute_stats.py — Compute per-channel mean and std over the training set

Run this once before training if you want to use your own computed stats
instead of the dataset-provided stats.yaml.

Usage
-----
python scripts/compute_stats.py \
    --data_dir /scratch/$USER/data/active_matter/data \
    --output stats_computed.yaml \
    --max_files 45
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import compute_channel_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default='/scratch/$USER/data/active_matter/data')
    parser.add_argument('--output',    default='stats_computed.yaml')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Limit files for a quick estimate (None = full train set)')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, 'train')
    print(f"Computing stats over: {train_dir}")

    stats = compute_channel_stats(
        data_dir     = train_dir,
        num_workers  = args.num_workers,
        max_files    = args.max_files,
    )

    with open(args.output, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)

    print(f"\nSaved → {args.output}")
