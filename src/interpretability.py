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


def save_feature_grid_visualization(
    top_features_dict: Dict[int, Dict[str, Any]], output_path: str
):
    """
    Generates and saves a unified 5x5 grid plot representing 5 SAE features (rows)
    and their top-5 activating spatial image crops (columns).

    Each row is labeled with its feature index and assigned CLIP concept.
    """
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    # ImageNet/CIFAR-10 normalization values to unnormalize crops for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    feature_indices = list(top_features_dict.keys())

    for r in range(5):
        if r >= len(feature_indices):
            for c in range(5):
                axes[r, c].axis("off")  # fill empty rows if we have fewer features
            continue

        feat_idx = feature_indices[r]
        feat_data = top_features_dict[feat_idx]
        exemplars = feat_data.get("exemplars", [])
        concept = feat_data.get("concept", "unknown")
        axes[r, 0].set_ylabel(
            f"Feature {feat_idx}\n({concept})",
            fontsize=9,
            fontweight="bold",
            rotation=0,
            labelpad=50,
            va="center",
            ha="right",
        )

        for c in range(5):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])

            if c >= len(exemplars):
                ax.axis("off")
                continue

            ex = exemplars[c]
            crop_tensor = ex["crop"]  # shape [3, H, W]
            act = ex["activation"]

            # channel-first tensor to channel-last numpy
            crop_np = crop_tensor.permute(1, 2, 0).cpu().numpy()

            # unnormalize
            crop_unnorm = crop_np * std + mean
            crop_unnorm = np.clip(crop_unnorm, 0.0, 1.0)

            ax.imshow(crop_unnorm)
            ax.set_title(f"Act: {act:.2f}", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Multi-feature exemplar grid saved to {output_path}")
