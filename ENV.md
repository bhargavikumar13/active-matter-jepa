# Environment Setup (ENV.md)

## Platform
NYU HPC Cloud Bursting via Open OnDemand: https://ood-burst-001.hpc.nyu.edu/
All jobs run inside a Singularity container with a Conda overlay.

## Singularity + Conda Setup

```bash
# 1. Copy an overlay template (15GB storage, 500K inodes)
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz /scratch/$USER/
gunzip /scratch/$USER/overlay-15GB-500K.ext3.gz

# 2. Launch a Singularity shell to create the environment
singularity exec --overlay /scratch/$USER/overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash

# 3. Inside the container — create the conda environment
source /ext3/miniconda3/etc/profile.d/conda.sh
conda create -n active_matter python=3.10 -y
conda activate active_matter

# 4. Install dependencies
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install h5py numpy pyyaml wandb matplotlib scikit-learn tqdm

# 5. Exit the container
exit
```

## Verifying the environment

```bash
singularity exec --nv \
    --overlay /scratch/$USER/overlay-15GB-500K.ext3:ro \
    /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/miniconda3/etc/profile.d/conda.sh && conda activate active_matter && python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
```

Expected output: `2.1.0 True`

## Data Location
```
/scratch/$USER/data/active_matter/
├── data/
│   ├── train/   # 45 HDF5 files, variable instances = 175 training trajectories
│   ├── valid/
│   └── test/
└── stats.yaml
```

## Downloading the dataset

```bash
singularity exec --overlay /scratch/$USER/overlay-15GB-500K.ext3:ro \
    /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate active_matter
    huggingface-cli download polymathic-ai/active_matter \
        --repo-type dataset \
        --local-dir /scratch/$USER/data/active_matter
    "
```

## WandB Setup

```bash
# Run once inside the container before training
wandb login
```

Or set the environment variable in your `.bashrc`:
```bash
export WANDB_API_KEY=your_key_here
```
