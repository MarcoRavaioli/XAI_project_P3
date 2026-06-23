import os

os.environ["MPLBACKEND"] = "agg"
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from model_loader import ActivationHook, ViTModelWrapper
from sae import SparseAutoencoder

logger = logging.getLogger(__name__)


def extract_patch_crop(
    image: torch.Tensor, spatial_idx: int, patch_size: int, grid_size: int
) -> torch.Tensor:
    """
    Extract the physical image crop corresponding to a spatial patch token index.
        row = (spatial_idx // grid_size) * patch_size
        col = (spatial_idx % grid_size) * patch_size

    Input:
        image [C, H, W]: Input image tensor.
        spatial_idx: Flat spatial index of the patch (0 to grid_size^2 - 1).
        patch_size: Width/height of each patch.
        grid_size: Number of patches along one dimension of the grid.
    Output:
        crop [C, patch_size, patch_size]: Cropped spatial patch tensor.
    """
    row = (spatial_idx // grid_size) * patch_size
    col = (spatial_idx % grid_size) * patch_size

    # assuming channel-first tensor layout [C, H, W]
    crop = image[:, row : row + patch_size, col : col + patch_size]
    return crop


def extract_contextual_crop(
    image: torch.Tensor,
    spatial_idx: int,
    patch_size: int,
    grid_size: int,
    context_patches: int = 2,
) -> torch.Tensor:
    """
    Extract a larger contextual crop around a spatial patch token index.
    Input:
        image [C, H, W]: Input image tensor.
        spatial_idx: Flat spatial index of the central patch.
        patch_size: Width/height of each patch.
        grid_size: Number of patches along one dimension of the grid.
        context_patches: Number of patches of padding on each side of the central patch.
    Output:
        crop [C, crop_h, crop_w]: Cropped context tensor.
    """
    row = spatial_idx // grid_size
    col = spatial_idx % grid_size

    row_start = max(0, row - context_patches) * patch_size
    row_end = min(grid_size, row + context_patches + 1) * patch_size
    col_start = max(0, col - context_patches) * patch_size
    col_end = min(grid_size, col + context_patches + 1) * patch_size

    return image[:, row_start:row_end, col_start:col_end]


def extract_activation_guided_crop(
    image: torch.Tensor,
    feature_map: Optional[torch.Tensor],
    patch_size: int,
    grid_size: int,
    context_patches: int = 2,
    activation_quantile: float = 0.9,
    min_active_tokens: int = 4,
    max_crop_tokens_per_side: int = 9,
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, int]]]:
    """
    Extract a crop centered on the high-activation region (not just a single max patch).
    Returns both crop tensor and the bounding box in image pixel coordinates.
    """
    if feature_map is None:
        return None, None

    fmap = feature_map.detach().cpu().numpy()
    if fmap.ndim == 1:
        if fmap.shape[0] != grid_size * grid_size:
            return None, None
        fmap = fmap.reshape(grid_size, grid_size)
    elif fmap.shape != (grid_size, grid_size):
        return None, None

    positive_vals = fmap[fmap > 0.0]
    if positive_vals.size == 0:
        return None, None

    q = float(np.quantile(positive_vals, activation_quantile))
    active_mask = fmap >= q

    # Keep at least a few active tokens to avoid degenerate 1-token crops.
    if int(active_mask.sum()) < int(max(1, min_active_tokens)):
        topk = min(int(max(1, min_active_tokens)), fmap.size)
        topk_idx = np.argpartition(fmap.reshape(-1), -topk)[-topk:]
        active_mask = np.zeros_like(fmap, dtype=bool)
        active_mask.reshape(-1)[topk_idx] = True

    rows, cols = np.where(active_mask)
    if len(rows) == 0 or len(cols) == 0:
        return None, None

    row_min = max(0, int(rows.min()) - context_patches)
    row_max = min(grid_size - 1, int(rows.max()) + context_patches)
    col_min = max(0, int(cols.min()) - context_patches)
    col_max = min(grid_size - 1, int(cols.max()) + context_patches)

    if max_crop_tokens_per_side > 0:
        peak_flat = int(np.argmax(fmap))
        peak_row = peak_flat // grid_size
        peak_col = peak_flat % grid_size

        def _cap_span(lo: int, hi: int, peak: int) -> Tuple[int, int]:
            span = hi - lo + 1
            if span <= max_crop_tokens_per_side:
                return lo, hi
            half = max_crop_tokens_per_side // 2
            new_lo = max(0, peak - half)
            new_hi = new_lo + max_crop_tokens_per_side - 1
            if new_hi >= grid_size:
                new_hi = grid_size - 1
                new_lo = max(0, new_hi - max_crop_tokens_per_side + 1)
            return new_lo, new_hi

        row_min, row_max = _cap_span(row_min, row_max, peak_row)
        col_min, col_max = _cap_span(col_min, col_max, peak_col)

    row_start = row_min * patch_size
    row_end = (row_max + 1) * patch_size
    col_start = col_min * patch_size
    col_end = (col_max + 1) * patch_size

    bbox = {
        "x": int(col_start),
        "y": int(row_start),
        "w": int(col_end - col_start),
        "h": int(row_end - row_start),
    }
    return image[:, row_start:row_end, col_start:col_end], bbox


def get_top_activating_patches(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    dataloader: DataLoader,
    layer_idx: int,
    feature_idx: int,
    k: int = 5,
    target_type: str = "mlp",
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Finds the top k patches across the dataset that trigger the highest activation
    for a specified SAE hidden feature index.

    Input:
        model_wrapper: Wrapped ViT/DINOv2.
        sae: Trained SparseAutoencoder.
        dataloader: Data loader of validation images.
        layer_idx: 0-indexed ViT layer depth.
        feature_idx: Index of target hidden feature in SAE.
        k: Number of exemplar patches to extract.
    """
    sae.to(device)
    sae.eval()

    submodule = model_wrapper.get_submodule(layer_idx, target_type)
    hook = ActivationHook(submodule)

    top_exemplars = []  # sorted list of top exemplars
    hook.register()
    logger.info(f"Searching for top-{k} exemplars for SAE Feature {feature_idx}...")

    # dynamically to support ViT vs DINOv2 differences
    patch_size = model_wrapper.patch_size
    grid_size = model_wrapper.grid_size

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch[0].to(device)

            _ = model_wrapper.model(images)

            # activation shape: [batch_size, seq_len, d_model]
            activation = hook.activation
            if activation is None:
                continue

            # Discard the global outlier [CLS] token at index 0.
            patch_tokens = activation[:, 1:, :]  # [batch_size, num_patches, d_model]
            f = sae.encode(patch_tokens)  # [batch_size, num_patches, hidden_dim]
            feature_activation = f[:, :, feature_idx]  # [batch_size, num_patches]

            # Identify high activations within the batch (taking the maximum activation per image)
            max_vals, max_idxs = feature_activation.max(dim=1)  # shapes: [batch_size]
            for b in range(images.shape[0]):
                img_tensor = batch[0][b]  # prevent memory leak
                max_act_val = max_vals[b].item()
                max_p = max_idxs[b].item()

                if max_act_val > 0.0:
                    crop = extract_patch_crop(img_tensor, max_p, patch_size, grid_size)
                    feature_map = feature_activation[b].detach().cpu()
                    label = batch[1][b].item() if len(batch) > 1 else 0
                    exemplar = {
                        "activation": max_act_val,
                        "crop": crop,
                        "full_image": img_tensor,
                        "feature_map": feature_map,
                        "spatial_idx": max_p,
                        "batch_idx": batch_idx,
                        "img_idx_in_batch": b,
                        "label": label,
                    }

                    top_exemplars.append(exemplar)
                    top_exemplars.sort(key=lambda x: x["activation"], reverse=True)
                    top_exemplars = top_exemplars[:k]

    hook.remove()
    logger.info(
        f"Found {len(top_exemplars)} activating exemplars. Max activation: {top_exemplars[0]['activation']:.4f}"
        if top_exemplars
        else "No activating exemplars found."
    )
    return top_exemplars


class CLIPAutoLabeler:
    """
    Zero-shot automatic visual feature labeling using CLIP.
    Computes cosine similarity between patch crops and candidate semantic labels.

    Citations:
        - Radford et al. [2021] ("Learning Transferable Visual Models From Natural Language Supervision")
    """

    def __init__(
        self, clip_model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"
    ):
        self.device = device
        logger.info(f"Loading CLIP Auto-Labeler ({clip_model_name})...")
        self.model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.model.eval()

    def label_feature(
        self,
        exemplars: List[Dict[str, Any]],
        candidate_concepts: List[str],
        patch_size: int = 16,
        grid_size: int = 14,
        context_patches: int = 2,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        top_k: int = 3,
        uncertainty_margin: float = 0.01,
        prompt_templates: List[str] = None,
        activation_crop_quantile: float = 0.9,
        activation_crop_min_tokens: int = 4,
        activation_crop_max_tokens: int = 9,
    ) -> Tuple[str, float, Dict[str, float], Dict[str, Any]]:
        """
        Assigns a semantic concept from candidates to the SAE feature using exemplar similarity.

        Input:
            exemplars: List of top exemplar dictionaries containing the image crop tensors.
            candidate_concepts: List of candidate labels.
        Output:
            assigned_concept: Either the best concept or 'uncertain' when margin is too small.
            best_score: Average similarity score for that concept.
            all_scores: Average similarity scores for all candidate concepts.
            metadata: Dict with raw best concept, uncertainty flag, margin, and top-k concepts.
        """
        if not exemplars:
            return (
                "inactive",
                0.0,
                {c: 0.0 for c in candidate_concepts},
                {
                    "best_raw_concept": "inactive",
                    "is_uncertain": False,
                    "margin_top1_top2": 0.0,
                    "top_concepts": [],
                },
            )

        if prompt_templates is None:
            prompt_templates = [
                "a photo of {concept}",
                "a close-up photo of {concept}",
                "an image featuring {concept}",
            ]

        # Use passed or default normalization values to unnormalize crops for CLIP
        if mean is None:
            mean = np.array([0.485, 0.456, 0.406])
        if std is None:
            std = np.array([0.229, 0.224, 0.225])

        # Use contextual crops centered around the active patch
        images_for_clip = []
        for ex in exemplars:
            if "full_image" in ex and "spatial_idx" in ex:
                crop_tensor, _ = extract_activation_guided_crop(
                    image=ex["full_image"],
                    feature_map=ex.get("feature_map"),
                    patch_size=patch_size,
                    grid_size=grid_size,
                    context_patches=context_patches,
                    activation_quantile=activation_crop_quantile,
                    min_active_tokens=activation_crop_min_tokens,
                    max_crop_tokens_per_side=activation_crop_max_tokens,
                )
                if crop_tensor is None:
                    crop_tensor = extract_contextual_crop(
                        image=ex["full_image"],
                        spatial_idx=ex["spatial_idx"],
                        patch_size=patch_size,
                        grid_size=grid_size,
                        context_patches=context_patches,
                    )
            else:
                crop_tensor = ex["crop"]

            img_np = crop_tensor.permute(1, 2, 0).cpu().numpy()
            img_unnorm = img_np * std + mean
            img_unnorm = np.clip(img_unnorm, 0.0, 1.0)
            img_uint8 = (img_unnorm * 255.0).clip(0, 255).astype(np.uint8)
            images_for_clip.append(img_uint8)

        # Process image crops using modern CLIPProcessor API signature with modality-specific kwargs
        image_inputs = self.processor(
            images=images_for_clip, images_kwargs={"return_tensors": "pt"}
        )
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        concept_prompt_pairs = []
        for concept in candidate_concepts:
            for template in prompt_templates:
                concept_prompt_pairs.append((concept, template.format(concept=concept)))

        formatted_texts = [prompt for _, prompt in concept_prompt_pairs]
        text_inputs = self.processor(
            text=formatted_texts, text_kwargs={"return_tensors": "pt", "padding": True}
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            # CLIP embeddings
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)

            if not isinstance(image_features, torch.Tensor):
                image_features = getattr(
                    image_features,
                    "image_features",
                    getattr(image_features, "pooler_output", image_features[0]),
                )
            if not isinstance(text_features, torch.Tensor):
                text_features = getattr(
                    text_features,
                    "text_features",
                    getattr(text_features, "pooler_output", text_features[0]),
                )

            # here we normalize embeddings to unit sphere
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity matrix: [num_exemplars, num_prompts]
            similarity_matrix = torch.matmul(image_features, text_features.t())

        sim_np = similarity_matrix.mean(dim=0).cpu().numpy()
        aggregated_scores: Dict[str, List[float]] = {c: [] for c in candidate_concepts}
        for idx, (concept, _) in enumerate(concept_prompt_pairs):
            aggregated_scores[concept].append(float(sim_np[idx]))

        all_scores = {
            concept: float(np.mean(scores)) if scores else 0.0
            for concept, scores in aggregated_scores.items()
        }

        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        best_concept, best_score = ranked[0]
        top_concepts = ranked[: max(1, top_k)]
        second_score = ranked[1][1] if len(ranked) > 1 else best_score
        margin = float(best_score - second_score)
        is_uncertain = margin < uncertainty_margin
        assigned_concept = "uncertain" if is_uncertain else best_concept

        logger.info(
            "Feature Auto-labeled as: "
            f"'{assigned_concept}' (best='{best_concept}', score={best_score:.4f}, margin={margin:.4f})"
        )

        metadata = {
            "best_raw_concept": best_concept,
            "is_uncertain": bool(is_uncertain),
            "margin_top1_top2": margin,
            "top_concepts": top_concepts,
        }
        return assigned_concept, float(best_score), all_scores, metadata


def get_feature_activation_map(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    image: torch.Tensor,
    layer_idx: int,
    feature_idx: int,
    target_type: str = "mlp",
    device: str = "cpu",
    activation_cache: Dict[Tuple[int, int], torch.Tensor] = None,
) -> np.ndarray:
    """
    Computes a 2D grid of feature activation intensities for a single image.
    Returns a numpy array of shape (grid_size, grid_size).
    """
    sae.eval()
    grid_size = model_wrapper.grid_size

    activation = None
    if activation_cache is not None:
        cache_key = (layer_idx, id(image))
        if cache_key in activation_cache:
            activation = activation_cache[cache_key]

    if activation is None:
        submodule = model_wrapper.get_submodule(layer_idx, target_type)
        hook = ActivationHook(submodule)
        hook.register()

        img_input = (
            image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
        )

        with torch.no_grad():
            _ = model_wrapper.model(img_input)
            activation = hook.activation
        hook.remove()

        if activation is None:
            logger.warning("No activation captured for heatmap.")
            return np.zeros((grid_size, grid_size))

        if activation_cache is not None:
            activation_cache[cache_key] = activation

    with torch.no_grad():
        patch_tokens = activation[:, 1:, :]
        f = sae.encode(patch_tokens)  # [1, 196, hidden_dim]
        feature_acts = f[0, :, feature_idx].detach().cpu().numpy()  # [196]

    return feature_acts.reshape(grid_size, grid_size)


def save_feature_grid_visualization(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    top_features_dict: Dict[int, Dict[str, Any]],
    output_path: str,
    layer_idx: int,
    patch_size: int = 16,
    grid_size: int = 14,
    device: str = "cpu",
    context_patches: int = 2,
):
    """
    Generates and saves a unified multi-feature grid plot.
    For each feature, it renders 3 rows:
      - Row 1: The 5 top-activating 16x16 patch crops (zoomed up).
      - Row 2: The full source images with the active patch highlighted.
      - Row 3: The activation heatmap overlay.
    """
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from PIL import Image as PILImage

    # Derive normalization constants from processor if available, otherwise use defaults
    processor = getattr(model_wrapper, "processor", None)
    mean_val = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std_val = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    mean = np.array(mean_val)
    std = np.array(std_val)

    feature_indices = list(top_features_dict.keys())
    num_features = len(feature_indices)

    # Initialize shared activation cache for all features being plotted in this layer grid
    activation_cache = {}

    fig, axes = plt.subplots(num_features * 3, 5, figsize=(15, num_features * 3 * 2.5))

    for r in range(num_features):
        feat_idx = feature_indices[r]
        feat_data = top_features_dict[feat_idx]
        exemplars = feat_data.get("exemplars", [])
        concept = feat_data.get("concept", "unknown")

        # Precompute heatmaps for all exemplars of this feature to find global_max and avoid duplicate forward passes
        heatmaps = []
        for ex in exemplars:
            if "full_image" in ex:
                hm = get_feature_activation_map(
                    model_wrapper=model_wrapper,
                    sae=sae,
                    image=ex["full_image"],
                    layer_idx=layer_idx,
                    feature_idx=feat_idx,
                    target_type="mlp",
                    device=device,
                    activation_cache=activation_cache,
                )
                heatmaps.append(hm)
            else:
                heatmaps.append(None)

        # Find global max activation for this feature's heatmaps (for consistent row-level color scaling)
        valid_maxes = [hm.max() for hm in heatmaps if hm is not None]
        global_max = max(valid_maxes) if valid_maxes else 1.0
        if global_max <= 0.0:
            global_max = 1.0

        # Set row labels on the left column (column 0)
        axes[r * 3 + 0, 0].set_ylabel(
            f"Feature {feat_idx}\n({concept})\n\nContext Crops",
            fontsize=10,
            fontweight="bold",
            rotation=0,
            labelpad=50,
            va="center",
            ha="right",
        )
        axes[r * 3 + 1, 0].set_ylabel(
            "Full Images",
            fontsize=10,
            fontweight="bold",
            rotation=0,
            labelpad=50,
            va="center",
            ha="right",
        )
        axes[r * 3 + 2, 0].set_ylabel(
            "Heatmaps",
            fontsize=10,
            fontweight="bold",
            rotation=0,
            labelpad=50,
            va="center",
            ha="right",
        )

        for c in range(5):
            # 1. Crops Row
            ax_crop = axes[r * 3 + 0, c]
            ax_crop.set_xticks([])
            ax_crop.set_yticks([])

            # 2. Full Image Row
            ax_full = axes[r * 3 + 1, c]
            ax_full.set_xticks([])
            ax_full.set_yticks([])

            # 3. Heatmap Row
            ax_heat = axes[r * 3 + 2, c]
            ax_heat.set_xticks([])
            ax_heat.set_yticks([])

            if c >= len(exemplars):
                ax_crop.axis("off")
                ax_full.axis("off")
                ax_heat.axis("off")
                continue

            ex = exemplars[c]
            act = ex["activation"]
            spatial_idx = ex.get("spatial_idx", None)
            activation_bbox = None

            # --- Row 1: Context Crop ---
            if "full_image" in ex and spatial_idx is not None:
                crop_tensor, activation_bbox = extract_activation_guided_crop(
                    image=ex["full_image"],
                    feature_map=ex.get("feature_map"),
                    patch_size=patch_size,
                    grid_size=grid_size,
                    context_patches=context_patches,
                )
                if crop_tensor is None:
                    crop_tensor = extract_contextual_crop(
                        image=ex["full_image"],
                        spatial_idx=spatial_idx,
                        patch_size=patch_size,
                        grid_size=grid_size,
                        context_patches=context_patches,
                    )
                    row_coord = spatial_idx // grid_size
                    col_coord = spatial_idx % grid_size
                    row_start_patch = max(0, row_coord - context_patches)
                    col_start_patch = max(0, col_coord - context_patches)
                    center_row_offset = (row_coord - row_start_patch) * patch_size
                    center_col_offset = (col_coord - col_start_patch) * patch_size
                else:
                    row_coord = spatial_idx // grid_size
                    col_coord = spatial_idx % grid_size
                    center_row_offset = (row_coord * patch_size) - activation_bbox["y"]
                    center_col_offset = (col_coord * patch_size) - activation_bbox["x"]
                
                crop_np = crop_tensor.permute(1, 2, 0).cpu().numpy()
                crop_unnorm = crop_np * std + mean
                crop_unnorm = np.clip(crop_unnorm, 0.0, 1.0)
                
                ax_crop.imshow(crop_unnorm, interpolation="nearest")
                
                # Highlight the exact central patch within the context crop
                rect = patches.Rectangle(
                    (center_col_offset, center_row_offset),
                    patch_size,
                    patch_size,
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none",
                )
                ax_crop.add_patch(rect)
            else:
                crop_tensor = ex["crop"]
                crop_np = crop_tensor.permute(1, 2, 0).cpu().numpy()
                crop_unnorm = crop_np * std + mean
                crop_unnorm = np.clip(crop_unnorm, 0.0, 1.0)
                ax_crop.imshow(crop_unnorm, interpolation="nearest")

            ax_crop.set_title(f"Act: {act:.2f}", fontsize=9)

            # --- Row 2: Full Image ---
            if "full_image" in ex:
                img_tensor = ex["full_image"]
                has_full = True
            else:
                img_tensor = ex["crop"]
                has_full = False

            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_unnorm = img_np * std + mean
            img_unnorm = np.clip(img_unnorm, 0.0, 1.0)
            ax_full.imshow(img_unnorm)

            if has_full and spatial_idx is not None:
                row_coord = (spatial_idx // grid_size) * patch_size
                col_coord = (spatial_idx % grid_size) * patch_size
                rect = patches.Rectangle(
                    (col_coord, row_coord),
                    patch_size,
                    patch_size,
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none",
                )
                ax_full.add_patch(rect)

                if activation_bbox is not None:
                    region_rect = patches.Rectangle(
                        (activation_bbox["x"], activation_bbox["y"]),
                        activation_bbox["w"],
                        activation_bbox["h"],
                        linewidth=1.5,
                        edgecolor="lime",
                        facecolor="none",
                    )
                    ax_full.add_patch(region_rect)

            # --- Row 3: Heatmap Overlay ---
            heatmap = heatmaps[c] if c < len(heatmaps) else None
            if has_full and heatmap is not None:
                heatmap_resized = (
                    np.array(
                        PILImage.fromarray(
                            (
                                Normalize(vmin=0.0, vmax=global_max)(heatmap) * 255
                            ).astype(np.uint8)
                        ).resize(
                            (img_unnorm.shape[1], img_unnorm.shape[0]),
                            PILImage.BILINEAR,
                        )
                    )
                    / 255.0
                )
                ax_heat.imshow(img_unnorm)
                ax_heat.imshow(heatmap_resized, cmap="hot", alpha=0.5)
            else:
                ax_heat.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Multi-feature exemplar grid saved to {output_path}")


def save_feature_activation_heatmap(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    image: torch.Tensor,
    layer_idx: int,
    feature_idx: int,
    target_type: str = "mlp",
    save_path: str = "feature_heatmap.png",
    device: str = "cpu",
):
    """
    Generates a spatial activation heatmap overlay for a single SAE feature on a source image.
    Each of the 196 patches gets colored by its feature activation intensity.
    """
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from PIL import Image as PILImage

    heatmap = get_feature_activation_map(
        model_wrapper=model_wrapper,
        sae=sae,
        image=image,
        layer_idx=layer_idx,
        feature_idx=feature_idx,
        target_type=target_type,
        device=device,
    )

    # unnormalize source image for display using derived constants
    processor = getattr(model_wrapper, "processor", None)
    mean_val = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std_val = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    mean = np.array(mean_val)
    std = np.array(std_val)
    img_np = image.cpu().numpy() if image.dim() == 3 else image[0].cpu().numpy()
    img_display = np.transpose(img_np, (1, 2, 0))
    img_display = img_display * std + mean
    img_display = np.clip(img_display, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(img_display)
    axes[0].set_title("Source Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="hot", interpolation="nearest")
    axes[1].set_title(
        f"Feature {feature_idx} Activation Map", fontsize=10, fontweight="bold"
    )
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_display)
    heatmap_resized = (
        np.array(
            PILImage.fromarray(
                (
                    Normalize(vmin=heatmap.min(), vmax=heatmap.max() + 1e-8)(heatmap)
                    * 255
                ).astype(np.uint8)
            ).resize((img_display.shape[1], img_display.shape[0]), PILImage.BILINEAR)
        )
        / 255.0
    )
    axes[2].imshow(heatmap_resized, cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(
        f"SAE Feature {feature_idx} — Spatial Activation Heatmap (Layer {layer_idx + 1})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature activation heatmap saved to {save_path}")
