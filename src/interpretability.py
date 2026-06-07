import os

os.environ["MPLBACKEND"] = "agg"
import logging
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple
import numpy as np
from transformers import CLIPModel, CLIPProcessor

from model_loader import ViTModelWrapper, ActivationHook
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

            # Identify high activations within the batch
            for b in range(images.shape[0]):
                img_tensor = batch[0][b]  # prevent memory leak
                for p in range(feature_activation.shape[1]):
                    act_val = feature_activation[b, p].item()
                    if act_val > 0.0:
                        crop = extract_patch_crop(img_tensor, p, patch_size, grid_size)
                        exemplar = {
                            "activation": act_val,
                            "crop": crop,
                            "full_image": img_tensor,
                            "spatial_idx": p,
                            "batch_idx": batch_idx,
                            "img_idx_in_batch": b,
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
        - Radhu et al. [2021] ("Do Vision Transformers See Like Convolutional Neural Networks?")
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
        self, exemplars: List[Dict[str, Any]], candidate_concepts: List[str]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Assigns a semantic concept from candidates to the SAE feature using exemplar similarity.

        Input:
            exemplars: List of top exemplar dictionaries containing the image crop tensors.
            candidate_concepts: List of candidate labels.
        Output:
            best_concept: Concept exhibiting the highest mean similarity.
            best_score: Average similarity score for that concept.
            all_scores: Average similarity scores for all candidate concepts.
        """
        if not exemplars:
            return "inactive", 0.0, {c: 0.0 for c in candidate_concepts}

        # ImageNet/CIFAR-10 normalization values to unnormalize crops for CLIP
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Use full source images for CLIP (tiny crops are too small for meaningful similarity)
        images_for_clip = []
        for ex in exemplars:
            if "full_image" in ex:
                img_np = ex["full_image"].permute(1, 2, 0).cpu().numpy()
            else:
                img_np = ex["crop"].permute(1, 2, 0).cpu().numpy()
            img_unnorm = img_np * std + mean
            img_unnorm = np.clip(img_unnorm, 0.0, 1.0)
            img_uint8 = (img_unnorm * 255.0).clip(0, 255).astype(np.uint8)
            images_for_clip.append(img_uint8)

        # Process image crops using modern CLIPProcessor API signature with modality-specific kwargs
        image_inputs = self.processor(
            images=images_for_clip, images_kwargs={"return_tensors": "pt"}
        )
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        formatted_texts = [f"a photo of {concept}" for concept in candidate_concepts]
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

            # Cosine similarity matrix: [num_exemplars, num_concepts]
            similarity_matrix = torch.matmul(image_features, text_features.t())
            mean_similarities = (
                similarity_matrix.mean(dim=0).cpu().numpy()
            )  # avg similarity

        all_scores = {
            concept: float(mean_similarities[idx])
            for idx, concept in enumerate(candidate_concepts)
        }
        best_idx = np.argmax(mean_similarities)
        best_concept = candidate_concepts[best_idx]
        best_score = float(mean_similarities[best_idx])

        logger.info(
            f"Feature Auto-labeled as: '{best_concept}' (score: {best_score:.4f})"
        )
        return best_concept, best_score, all_scores


def get_feature_activation_map(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    image: torch.Tensor,
    layer_idx: int,
    feature_idx: int,
    target_type: str = "mlp",
    device: str = "cpu",
) -> np.ndarray:
    """
    Computes a 2D grid of feature activation intensities for a single image.
    Returns a numpy array of shape (grid_size, grid_size).
    """
    sae.eval()
    grid_size = model_wrapper.grid_size

    submodule = model_wrapper.get_submodule(layer_idx, target_type)
    hook = ActivationHook(submodule)
    hook.register()

    img_input = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)

    with torch.no_grad():
        _ = model_wrapper.model(img_input)
        activation = hook.activation
    hook.remove()

    if activation is None:
        logger.warning("No activation captured for heatmap.")
        return np.zeros((grid_size, grid_size))

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
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import Normalize
    from PIL import Image as PILImage

    # ImageNet/CIFAR-10 normalization values to unnormalize images for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    feature_indices = list(top_features_dict.keys())
    num_features = len(feature_indices)

    fig, axes = plt.subplots(num_features * 3, 5, figsize=(15, num_features * 3 * 2.5))

    for r in range(num_features):
        feat_idx = feature_indices[r]
        feat_data = top_features_dict[feat_idx]
        exemplars = feat_data.get("exemplars", [])
        concept = feat_data.get("concept", "unknown")

        # Set row labels on the left column (column 0)
        axes[r * 3 + 0, 0].set_ylabel(
            f"Feature {feat_idx}\n({concept})\n\nCrops",
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

            # --- Row 1: Crop ---
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

            # --- Row 3: Heatmap Overlay ---
            if has_full:
                heatmap = get_feature_activation_map(
                    model_wrapper=model_wrapper,
                    sae=sae,
                    image=img_tensor,
                    layer_idx=layer_idx,
                    feature_idx=feat_idx,
                    target_type="mlp",
                    device=device,
                )
                heatmap_resized = (
                    np.array(
                        PILImage.fromarray(
                            (
                                Normalize(
                                    vmin=heatmap.min(), vmax=heatmap.max() + 1e-8
                                )(heatmap)
                                * 255
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

    # unnormalize source image for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
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
