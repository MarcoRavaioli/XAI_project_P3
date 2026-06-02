# Mechanistic Interpretability of Vision Transformers using Sparse Autoencoders

## Project Goals

This project implements a modular pipeline for research on **Mechanistic Interpretability of Vision Transformers (ViTs) using Sparse Autoencoders (SAEs)**. The pipeline is designed to dissect the internal representations of pre-trained ViT models (both supervised and self-supervised DINOv2) to resolve two core research challenges:

1.  **The Spatial Challenge**: Mapping high-dimensional internal activations of spatial patch tokens to visual crops and analyzing their causal contribution to the final target class prediction boundaries.
2.  **Cross-Modal Evaluation**: Identifying what specific visual features stand for by automatically labeling top-activating spatial crops using CLIP and verifying their semantic alignment.

Docstrings and mathematical modules are cited against foundational interpretability and ViT literature:

- **SAE & Monosemanticity**: _Bricken et al. [2023]_ ("Towards Monosemanticity"), _Cunningham et al. [2024]_ ("Sparse Autoencoders Find Highly Interpretable Features").
- **ViT Mechanics**: _Dosovitskiy et al. [2021]_ ("An Image is Worth 16x16 Words"), _Raghu et al. [2021]_ ("Do Vision Transformers See Like Convolutional Neural Networks?").
- **Transformer Circuits**: _Elhage et al. [2021]_ ("A Mathematical Framework for Transformer Circuits").

---

## Modules

### 1. `model_loader.py` (ViT Backbone & Activation Caching)

- Loads pre-trained paradigms: Supervised ViT (`google/vit-base-patch16-224`) and Self-Supervised DINOv2 (`facebook/dinov2-base`).
- Dynamically configures backbone patch sizes ($16\times16$ vs $14\times14$) and grid divisions ($14\times14$ vs $16\times16$) to ensure shape alignment.
- Resolves structural differences across Hugging Face `transformers` versions (handling direct `layers` root container arrays or standard nested `encoder.layer` submodules).
- Features a reusable forward `ActivationHook` class for activation caching and intervention injection.

### 2. `sae.py` (Sparse Autoencoder Architecture)

- Implements the `SparseAutoencoder` module representing $W_{enc} \in \mathbb{R}^{d \times m}$ and $W_{dec} \in \mathbb{R}^{m \times d}$ (expansion factor $F=8$).
- Applies a pre-bias centering subtraction ($x - b_{dec}$) before encoding as described in state-of-the-art architectures (_Bricken et al. [2023]_).
- Includes a post-update projection method to normalize decoder column weights to unit $L_2$ norm ($||W_{dec, j}||_2 = 1.0$) to avoid L1 optimization shrinkage shortcuts.
- Computes combined loss: $\text{Loss} = \text{MSE}(x, \hat{x}) + \lambda \sum |f(x)|_1$.

### 3. `caching_and_training.py` (Data Pipeline & Training Loop)

- Implements a streaming **`TokenActivationBuffer`** to buffer patch activations batch-by-batch from the DataLoader.
- **Outlier Discarding**: Strips the outlier global classification token (`[CLS]` at index `0`) to force the SAE to focus strictly on localized visual representations.
- Tracks reconstruction fidelity ($R^2$ Score) and activation sparsity ($L_0$ Norm) across updates.
- Includes scaling device auditing that issues warning logs if a CUDA-enabled GPU is available but fallback is configured on CPU.

### 4. `interpretability.py` (Exemplar Search & Auto-Labeling)

- Implements exemplar patch search, mapping flat patch activation indices to spatial row/column grid crops based on the specific backbone dimension.
- Leverages `CLIPAutoLabeler` to query pre-trained CLIP models (`openai/clip-vit-base-patch32`) to compute cosine similarities between visual crop sets and textual candidate concepts.
- **Exemplar Grid Visualization**: Generates a high-resolution $5 \times 5$ matrix plot showcasing 5 representative features (rows) and their top-5 spatial patch exemplars (columns), unnormalizing them back to clean RGB format.

### 5. `causal_eval.py` (Causal Interventions)

- **Surgical Ablation Hook**: Intercepts the forward pass, isolates the `[CLS]` token, zero-activates the target feature in SAE space, and surgically subtracts the feature projection vector directly from the residual stream to bypass reconstruction noise:
  $$x_{\text{ablated}} = x_{\text{patches}} - f_j(x) W_{dec, j}$$
- **Surgical Steering Hook**: Surgically injects scaling interventions on targeted representations:
  $$x_{\text{steered}} = x_{\text{patches}} + (S - 1) f_j(x) W_{dec, j}$$
- Computes Relative Logit Drop and plots dose-response curves.

### 6. `run_pipeline.py` (Integration Runner)

- Connects all modules into a unified multi-layer comparative loop iterating through **Layer 6 (Mid-Network)** and **Layer 11 (Late-Network)**.
- Exports quantitative tabular tables and CSV summaries.

---

## Getting Started & Usage

This project uses `uv` for package management.

### 1. Installation

Initialize the virtual environment and fetch all dependencies:

```bash
uv sync
```

### 2. Test Run

```bash
# quick verification loop
uv run python run_pipeline.py --subset_size 10 --epochs 1
```

### 3. Full Run

```bash
# CUDA with larger subset size and multiple epochs
uv run python run_pipeline.py --device cuda --subset_size 500 --epochs 5
```

---

## Artifacts

Each execution of the comparative loop generates four artifacts under the root directory:

### 1. `layer_comparison_summary.md`

A Markdown-formatted comparison table evaluating layers at different depths.
_Example output_:
| Layer | R^2 Score | L_0 Norm | Mean Logit Drop |
|------------|-----------|----------|-------------------|
| Layer 6 | 0.0379 | 2382.81 | 0.7978% |
| Layer 11 | -0.4434 | 1845.31 | -0.0986% |

### 2. `multi_feature_exemplar_grid.png`

A unified figure showcasing a $5 \times 5$ matrix of 5 representative features (rows) and their top-5 spatial image crops (columns). Rows are explicitly labeled with their index and assigned CLIP concept (e.g. `"Feature 4049 (red color)"`).

### 3. `discovered_features_summary.csv`

A detailed quantitative exporter file tracking discovered representation statistics across layers.
_Columns mapped_:

- `Feature Index`: Flat SAE index.
- `Target Layer`: Depth evaluation (e.g., `Layer 6`).
- `Assigned CLIP Concept`: High-similarity concept returned by CLIP Auto-Labeler.
- `CLIP Confidence Score`: Averaged cosine similarity margin.
- `Baseline Logit`: Classifier output before intervention.
- `Ablated Logit`: Logit after surgical feature zeroing.
- `Relative Logit Drop (%)`: Causal impact metric of feature deletion.
- `Steered Logit Increase (%)`: Logit shift margin under $5.0\times$ feature scaling.

### 4. `dose_response_curve.png`

A dose-response causal graph showing the Relative Logit Drop (%) as a function of the ablation strength (varying from 0% to 100% ablation).
