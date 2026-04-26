# Active Matter — Self-Supervised Representation Learning

Self-supervised representation learning on the ⁠ active_matter ⁠ dataset from
[The Well](https://github.com/PolymathicAI/the_well) using a
Joint-Embedding Predictive Architecture (JEPA).

Our best model achieves a normalised test MSE of *0.095* via linear probing,
outperforming an end-to-end supervised baseline (0.134), suggesting that
SSL representations can generalise better than supervised training on this small,
label-scarce physical simulation dataset.

---

## Results

| Method | Combined MSE | α MSE | ζ MSE |
|---|---|---|---|
| Random baseline | 1.000 | 1.000 | 1.000 |
| kNN (k=20), Run 1 — 3.5M, 135 samples | 0.869 | 0.683 | 1.056 |
| Linear probe, Run 1 | 0.719 | 0.665 | 0.774 |
| kNN (k=10), Run 2 — 3.5M, 875 samples | 0.341 | 0.105 | 0.577 |
| Linear probe, Run 2 | 0.270 | 0.101 | 0.438 |
| kNN (k=20), Run 3 — 3.5M, 11,550 samples | 0.257 | 0.065 | 0.448 |
| Linear probe, Run 3 | 0.185 | 0.048 | 0.323 |
| kNN (k=20), Run 4 — 26.6M, harder masking | 0.424 | 0.172 | 0.676 |
| Linear probe, Run 4 | 0.316 | 0.109 | 0.524 |
| kNN (k=20), Run 5 — 26.6M, 11,550 samples | 0.258 | 0.117 | 0.400 |
| Separate probes, Run 5 | 0.090 | 0.039 | 0.142 |
| Epoch ensemble (ep80+90+best), Run 5 | 0.091 | 0.039 | 0.144 |
| *Linear probe, Run 5 (official)* | *0.095* | *0.038* | *0.152* |
| Attention pooling probe, Run 5 (exploratory) | 0.088 | 0.035 | 0.141 |
| Supervised baseline | 0.134 | 0.229 | 0.039 |

All MSE values are on normalised labels (z-score). Lower is better.
kNN regression uses cosine similarity in frozen embedding space.

	⁠*Official evaluation:* Linear probing (0.095) and kNN (0.258) only, in compliance with
	⁠project requirements. Attention pooling and ensemble methods are exploratory analyses, not used for official evaluation.
>
	⁠*Run 4 kNN (0.424)* is worse than Run 3 (0.257) despite 7.6× more parameters — this likely
	⁠reflects objective misalignment rather than simple overfitting. Aggressive masking
	⁠may have optimised local prediction at the expense of global parameter-predictive structure.

---

## Dataset

Simulation of rod-like active particles immersed in a Stokes fluid. Each
trajectory is governed by two physical parameters:

•⁠  ⁠*α* (active dipole strength): {-1, -2, -3, -4, -5} — 5 discrete values
•⁠  ⁠*ζ* (steric alignment): {1, 3, 5, 7, 9, 11, 13, 15, 17} — 9 discrete values

45 unique parameter combinations × variable trajectories per combination.

| Property | Value |
|---|---|
| Trajectories | 175 train / 24 val / 26 test |
| Time steps | 81 per trajectory |
| Spatial resolution | 256×256 (cropped to 224×224) |
| Physical channels | 11 |
| Training samples (stride-1) | 11,550 train / 1,584 val / 1,716 test |
| Total size | ~52 GB |

### Physical channels

| Count | Field | Type |
|---|---|---|
| 1 | Concentration | Scalar field |
| 2 | Velocity (vx, vy) | Vector field |
| 4 | Orientation tensor D | 2×2 tensor, flattened |
| 4 | Strain-rate tensor E | 2×2 tensor, flattened |
| *11* | *Total* | |

---

## Model Architecture

### JEPA overview

Unlike reconstruction-based methods (e.g., MAE), JEPA predicts latent
representations rather than raw pixels, encouraging the model to learn
higher-level structure and dynamics instead of low-level reconstruction.
Unlike contrastive methods (e.g., SimCLR), no data augmentation is required —
which is important for physical fields where augmentations have no natural
physical analogue.



Input clip (16, 11, 224, 224)
        │
        ▼
  Tubelet embedding
  (t=2, h=16, w=16) → 1568 tokens
        │
   ┌────┴─────────────────┐
   │                       │
Context encoder         Target encoder
  (masked context)       (EMA copy, no grad)
   │                       │
   └──────► Predictor ◄────┘
            (predicts target
             embeddings from
             masked positions)


### Parameter counts

| Configuration | Encoder | Predictor | Total |
|---|---|---|---|
| Small (Runs 1–3) | 2.86M | 0.64M | 3.50M |
| Large (Runs 4–5) | 23.46M | 3.12M | *26.58M* |

Both configurations are well under the 100M parameter limit.

### Key design choices

•⁠  ⁠*3D tubelet embeddings* — ⁠ (t=2, h=16, w=16) ⁠ patches capture spatiotemporal structure
•⁠  ⁠*Spatiotemporal block masking* — 4 contiguous 3D target regions (15–30% of tokens each)
•⁠  ⁠*EMA target encoder* — prevents representation collapse without contrastive pairs
•⁠  ⁠*Mean pooling* — all 1568 tokens averaged for linear probe / kNN features
•⁠  ⁠*No pretrained weights* — all models trained from scratch on active_matter only

---

## Directory Structure


active_matter/
├── src/
│   ├── utils.py                      # Shared utilities: DotDict, load_config, resolve_paths
│   ├── dataset.py                    # PyTorch Dataset + DataLoader factory
│   ├── model.py                      # 3D ViT encoder, JEPA predictor, JEPA model
│   └── masking.py                    # Spatiotemporal block masking
├── scripts/
│   ├── train.py                      # SSL pre-training
│   ├── probe.py                      # Linear probing evaluation
│   ├── eval_knn.py                   # kNN regression evaluation
│   ├── supervised.py                 # End-to-end supervised baseline
│   ├── compute_stats.py              # Per-channel mean/std computation
│   ├── generate_submission.py        # Submission CSV generator (experimental)
│   ├── probe_cv.py                   # 5-fold cross-validation probe (experimental)
│   ├── probe_sweep.py                # L2 regularization sweep (experimental)
│   ├── probe_separate.py             # Separate probes for α and ζ (experimental)
│   ├── probe_ensemble_checkpoints.py # Epoch ensemble probe (experimental)
│   ├── attention_pool_probe.py       # Attention pooling probe (experimental)
│   └── visualize_embeddings.py       # UMAP/t-SNE embedding visualization
├── configs/
│   ├── jepa.yaml                     # Default config (Run 5, best model)
│   ├── jepa_run3.yaml                # Run 3 — 3.5M, full dataset
│   ├── jepa_run4.yaml                # Run 4 — 26.6M, harder masking
│   └── jepa_run5.yaml                # Run 5 — 26.6M, easier masking (best)
├── slurm/
│   ├── train.sbatch                  # GPU pre-training job
│   ├── probe.sbatch                  # Linear probe + kNN evaluation job
│   ├── ablations.sbatch              # Ablation studies job
│   ├── eval.sbatch                   # Full evaluation pipeline job
│   ├── supervised.sbatch             # Supervised baseline job
│   ├── submission.sbatch             # Submission generation job (experimental)
│   ├── competition.sbatch            # All competition scripts (experimental)
│   ├── knn_run3.sbatch               # kNN evaluation for Run 3
│   ├── knn_run4.sbatch               # kNN evaluation for Run 4
│   └── viz.sbatch                    # Embedding visualization job
├── figures/
│   └── embedding_viz.png            # UMAP embedding visualization (generated)
├── explore/
│   └── inspect_hdf5.py              # Data exploration script
├── eval.sh                           # End-to-end evaluation script
├── run.sh                            # Singularity container helper
├── monitor.sh                        # Auto-requeue training monitor
├── stats.yaml                        # Per-channel normalisation statistics
├── active_matter.yaml                # Dataset metadata from The Well
├── requirements.txt                  # Python dependencies
├── ENV.md                            # Environment setup guide
├── visualization_active_matter.ipynb # Dataset exploration notebook
└── README.md                            # This file


Scripts marked (experimental) are prepared for potential future competition
use and are not part of the main course evaluation pipeline.

---

## Environment Setup

See [ENV.md](ENV.md) for full Singularity + Miniconda setup instructions
including overlay creation, conda environment setup, and package installation.

Quick check — enter the container:

⁠ bash
bash run.sh
 ⁠

---

## Reproducing Results

### 0. Set up your data path

All config files use ⁠ $USER ⁠ for paths (e.g. ⁠ /scratch/$USER/data/active_matter/ ⁠).
This is expanded automatically at runtime — no manual editing required.
Just make sure your data lives at:

/scratch/$USER/data/active_matter/data/


### 1. WandB setup (optional but recommended)

Training logs metrics to [Weights & Biases](https://wandb.ai). To enable:

⁠ bash
# Inside the container
wandb login
# Or set in your environment
export WANDB_API_KEY=your_key_here
 ⁠

To disable WandB entirely, add ⁠ --no-wandb ⁠ to any training command. The
sbatch files use ⁠ YOUR_WANDB_API_KEY ⁠ as a placeholder — replace it with
your key or remove the export line and use ⁠ --no-wandb ⁠.

### 2. Pre-training (Run 5 — best model)

⁠ bash
sbatch slurm/train.sbatch
 ⁠

Or directly:

⁠ bash
bash run.sh python scripts/train.py --config configs/jepa_run5.yaml
 ⁠

### 3. Evaluation

⁠ bash
bash eval.sh checkpoints/jepa/best.pt configs/jepa_run5.yaml
 ⁠

This runs linear probing and kNN regression and saves results to
⁠ checkpoints/jepa/probe/ ⁠.

Or step by step:

⁠ bash
# Linear probe
bash run.sh python scripts/probe.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    --no-wandb

# kNN regression
bash run.sh python scripts/eval_knn.py \
    --config configs/jepa_run5.yaml \
    --checkpoint checkpoints/jepa/best.pt \
    --k 1 5 10 20 50 \
    --no-wandb
 ⁠

Results saved to ⁠ checkpoints/jepa/probe/ ⁠:
•⁠  ⁠⁠ probe_results__best__mean.yaml ⁠ — linear probing MSE
•⁠  ⁠⁠ knn_results__best__mean.yaml ⁠ — kNN regression MSE across k values

### 4. Supervised baseline

⁠ bash
sbatch slurm/supervised.sbatch
 ⁠

### 5. Sanity check (CPU, no GPU needed)

⁠ bash
bash run.sh python scripts/train.py --config configs/jepa.yaml \
    training.epochs=2 \
    training.batch_size=2 \
    training.val_batches=2 \
    training.num_workers=2 \
    training.use_amp=false \
    data.spatial_size=64 \
    --no-wandb
 ⁠

---

## Scaling Analysis

| Run | Samples | Params | Val loss | LP MSE | kNN MSE |
|---|---|---|---|---|---|
| Run 1 | 135 | 3.5M | 0.1417 | 0.719 | 0.869 |
| Run 2 | 875 | 3.5M | 0.0613 | 0.270 | 0.341 |
| Run 3 | 11,550 | 3.5M | 0.0680 | 0.185 | 0.257 |
| Run 4 | 11,550 | 26.6M | 0.0424 | 0.316 | 0.424 |
| Run 5 | 11,550 | 26.6M | 0.0699 | *0.095* | *0.258* |

Key insight: JEPA val loss is not a reliable proxy for downstream LP or kNN MSE.
Run 4 achieves the best val loss (0.0424) but worst performance on both LP (0.316)
and kNN (0.424) — suggesting objective misalignment rather than simple overfitting.
Aggressive masking may preferentially optimise local spatiotemporal prediction at the expense
of global parameter-predictive structure.

---

## Probing Strategy Analysis

Additional experiments on the Run 5 encoder comparing different evaluation strategies:

| Method | Combined MSE | α MSE | ζ MSE | Notes |
|---|---|---|---|---|
| Joint linear probe | 0.095 | 0.038 | 0.152 | Default evaluation |
| Separate probes (α, ζ) | 0.090 | 0.039 | 0.142 | Independent probes per parameter |
| L2 sweep (wd=1e-5) | 0.0935 | 0.039 | 0.149 | Default wd=1e-4 nearly optimal |
| 5-fold CV ensemble | 0.0951 | 0.041 | 0.149 | Stable across folds (±0.005) |
| Epoch ensemble (80+90+best) | 0.0913 | 0.039 | 0.144 | ep90 best single (0.0913) |
| *Attention pooling probe* | *0.0879* | *0.035* | *0.141* | *Best overall result* |

Key findings:
•⁠  ⁠Attention pooling (0.0879) outperforms mean pooling (0.095) by 7.5%, particularly for ζ (0.141 vs 0.152), suggesting the encoder produces spatially differentiated representations where selective token weighting may recover locally-encoded information that mean pooling averages out.
•⁠  ⁠Separate probes improve ζ from 0.152 → 0.142, suggesting α and ζ encode largely independent structure that may compete when predicted jointly.
•⁠  ⁠Default weight decay (1e-4) is nearly optimal — 1e-5 gives marginal improvement (0.0935 vs 0.0936), indicating the representations are naturally well-regularized.
•⁠  ⁠CV variance is low (±0.005), confirming results are robust across data splits.

---

## Reproducibility

All experiments use:
•⁠  ⁠Seed: 42
•⁠  ⁠⁠ torch.backends.cudnn.deterministic = True ⁠
•⁠  ⁠⁠ torch.backends.cudnn.benchmark = False ⁠
•⁠  ⁠Mixed precision (AMP): enabled

---

## Team

| Name | NetID | Contribution |
|---|---|---|
| Bhargavi Priyasha Kumar | bk3228 | Training pipeline (mixed precision, LR scheduling, checkpoint management), linear probing, kNN regression evaluation, HPC infrastructure |
| Vikash Agarwal | va2661 | 3D ViT encoder, JEPA predictor, EMA target encoder, spatiotemporal masking, ablation studies |
| Mayank Garg | mg8948 | HDF5 dataset pipeline, temporal windowing, per-channel normalisation, DataLoader infrastructure, supervised baseline, report writing |

---

## Citation

⁠ bibtex
@article{maddu2024learning,
  title={Learning fast, accurate, and stable closures of a kinetic theory of an active fluid},
  author={Maddu, Suryanarayana and Weady, Scott and Shelley, Michael J},
  journal={Journal of Computational Physics},
  volume={504},
  pages={112869},
  year={2024},
  publisher={Elsevier}
}

@article{ohana2024well,
  title={The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning},
  author={Ohana, Ruben and others},
  journal={arXiv preprint arXiv:2412.00568},
  year={2024}
}

@inproceedings{assran2023ijepa,
  title={Self-supervised learning from images with a joint-embedding predictive architecture},
  author={Assran, Mahmoud and others},
  booktitle={CVPR},
  year={2023}
}
 ⁠
