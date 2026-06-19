# Mechanistic Interpretability of Vision Transformers using Sparse Autoencoders

This repository contains the code and paper for a project on **mechanistic interpretability of Vision Transformers (ViTs) using Sparse Autoencoders (SAEs)**, developed for the _Explainable and Trustworthy AI_ course at Politecnico di Torino.

The pipeline trains SAEs on intermediate MLP activations of a pre-trained ViT-B/16, extracts sparse feature directions, labels them automatically via CLIP zero-shot similarity, and evaluates their causal role through surgical ablation and steering interventions. Two network depths are compared: Layer 6 (mid-network) and Layer 11 (late-network).

## Modules

### `model_loader.py`

Loads pre-trained backbones: supervised ViT (`google/vit-base-patch16-224`) and self-supervised DINOv2 (`facebook/dinov2-base`). Dynamically resolves patch size (16×16 vs 14×14) and grid dimensions (14×14 vs 16×16) across model families. Provides a reusable `ActivationHook` class for both activation caching and forward-pass intervention injection (used by the causal evaluation module).

### `sae.py`

Implements a one-hidden-layer Sparse Autoencoder with expansion factor $F{=}8$ ($m{=}6144$ hidden units for $d{=}768$). Applies pre-bias centering ($x - b_\text{dec}$) before encoding and adds $b_\text{dec}$ back after decoding, following Bricken et al. [2023]. Decoder columns are projected to unit $L_2$ norm after each optimiser step to prevent $L_1$ shrinkage shortcuts. Loss is $\text{MSE}(x, \hat{x}) + \lambda \|f(x)\|_1$.

### `caching_and_training.py`

Implements a streaming `TokenActivationBuffer` that feeds patch-token activations to the SAE batch-by-batch, discarding the `[CLS]` token (index 0) so the SAE learns only from spatially localised representations. The buffer shuffles across batch boundaries to reduce intra-batch correlation. The training loop tracks $R^2$ (reconstruction fidelity) and $L_0$ (mean number of active features per token).

### `interpretability.py`

Searches for the top-$k$ highest-activating patches per SAE feature across the dataset. A `CLIPAutoLabeler` computes mean cosine similarity between contextual crops around each exemplar patch and a set of candidate concept prompts (`"a photo of {concept}"`), assigning the best-matching label. The grid visualisation renders three rows per feature: (1) contextual crops with the central patch highlighted, (2) full source images with the active patch outlined, and (3) spatial activation heatmap overlays.

### `causal_eval.py`

Implements surgical causal interventions that operate directly in the residual stream:

- **Ablation**: $x' = x - (1-s) \cdot f_j(x) \cdot W_{\text{dec},j}$ where $s{=}0$ is full ablation.
- **Steering**: $x' = x + (S-1) \cdot f_j(x) \cdot W_{\text{dec},j}$ where $S{=}5$ amplifies the feature 5×.

The `[CLS]` token is preserved untouched during intervention. Relative Logit Drop (RLD) quantifies causal impact; dose-response curves sweep ablation strength from 0% to 100%.

### `run_pipeline.py`

Connects all modules into a multi-layer comparative loop over Layer 6 and Layer 11. For each layer: trains an SAE, identifies the 10 most active features, labels them via CLIP, runs ablation and steering on each, and selects the top 5 (by CLIP confidence) for grid visualisation. Exports all quantitative results as CSV and Markdown.

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. All commands should be run from `src/`.

```bash
cd src
uv sync
```

### Quick test

```bash
uv run python run_pipeline.py --subset_size 10 --epochs 1
```

### Full run (as reported in the paper)

```bash
uv run python run_pipeline.py \
    --dataset imagenet --subset_size 1000 \
    --epochs 5 15 --l1_coeff 1e-3 \
    --device cuda
```

## CLI Arguments

| Argument            | Default                       | Description                                                         |
| ------------------- | ----------------------------- | ------------------------------------------------------------------- |
| `--model`           | `google/vit-base-patch16-224` | Backbone (`google/vit-base-patch16-224` or `facebook/dinov2-base`)  |
| `--dataset`         | `cifar10`                     | Dataset: `cifar10`, `imagewoof`, `imagenette`, `imagenet`           |
| `--dataset_path`    | `./data`                      | Local directory for dataset storage                                 |
| `--subset_size`     | `50`                          | Number of validation images to use                                  |
| `--epochs`          | `1`                           | Training epochs per layer (space-separated for multiple layers)     |
| `--l1_coeff`        | `1e-3`                        | $L_1$ sparsity coefficient (space-separated for multiple layers)    |
| `--expansion`       | `8`                           | SAE overcomplete expansion ratio                                    |
| `--feature_idx`     | `100`                         | SAE feature index to analyse                                        |
| `--eval_image_idx`  | `0`                           | Index of the evaluation image for causal interventions              |
| `--context_patches` | `2`                           | Patches of spatial context around each exemplar crop (2 → 5×5 crop) |
| `--device`          | `cpu`                         | Compute device (`cpu` or `cuda`)                                    |

ImageWoof and ImageNette are auto-downloaded if not present. ImageNet-100 falls back to the HuggingFace `clane9/imagenet-100` dataset if no local copy exists (requires `pip install datasets`).

## Output Artifacts

All outputs are written to `src/out/`.

| File                                       | Description                                                                                              |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `layer_comparison_summary.md`              | Markdown table comparing $R^2$, $L_0$, and mean RLD across layers                                        |
| `discovered_features_summary.csv`          | Per-feature quantitative results (CLIP label, confidence, baseline/ablated logits, RLD, steering effect) |
| `multi_feature_exemplar_grid_layer{N}.png` | Grid of top-5 features (by CLIP score) at layer $N$, with context crops, full images, and heatmaps       |
| `feature_grid_layer{N}_feat{M}.png`        | Single-feature grid for feature $M$ at layer $N$                                                         |
| `dose_response_curve.png`                  | RLD as a function of ablation strength for the most causally active Layer 11 feature                     |
| `feature_activation_heatmap.png`           | Spatial activation map for the most causally active Layer 11 feature                                     |
| `sae_training_curves.png`                  | Loss, $R^2$, and $L_0$ convergence curves for both layers                                                |

## References

- Bricken et al. [2023], "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
- Cunningham et al. [2024], "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
- Dosovitskiy et al. [2021], "An Image is Worth 16x16 Words"
- Elhage et al. [2021], "A Mathematical Framework for Transformer Circuits"
- Raghu et al. [2021], "Do Vision Transformers See Like Convolutional Neural Networks?"
- Radford et al. [2021], "Learning Transferable Visual Models From Natural Language Supervision"
