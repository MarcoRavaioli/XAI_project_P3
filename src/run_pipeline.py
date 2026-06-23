import os
import re

os.environ["MPLBACKEND"] = "agg"
import argparse
import csv
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import InterpolationMode

from caching_and_training import TokenActivationBuffer, train_sae
from causal_eval import (
    evaluate_causal_dataset,
    evaluate_relative_logit_drop,
    plot_dose_response,
    plot_training_curves,
)
from interpretability import (
    CLIPAutoLabeler,
    get_top_activating_patches,
    save_feature_activation_heatmap,
    save_feature_grid_visualization,
)
from model_loader import ActivationHook, ViTModelWrapper, get_num_classes, set_seed
from sae import SparseAutoencoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("run_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mechanistic Interpretability Pipeline with Sparse Autoencoders"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/vit-base-patch16-224",
        choices=["google/vit-base-patch16-224", "facebook/dinov2-base"],
        help="Hugging Face model checkpoint backbone",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 3, 7, 10],
        help="0-indexed layer indices to compare (e.g. 1 3 7 10 corresponds to Layer 2, 4, 8, 11)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[1],
        help="Number of training epochs for SAE (can specify multiple values mapped to each layer)",
    )
    parser.add_argument(
        "--l1_coeff",
        type=float,
        nargs="+",
        default=[1e-3],
        help="L1 coefficient scaling weight of feature sparsity (can specify multiple values mapped to each layer)",
    )
    parser.add_argument(
        "--expansion",
        type=int,
        default=8,
        help="Overcomplete expansion ratio (hidden_dim = F * d_model)",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=1000,
        help="Size of subset for training and evaluation. Values below ~200 are for fast iteration/smoke-testing only, not for final results",
    )
    parser.add_argument(
        "--feature_idx",
        type=int,
        default=None,
        help="SAE hidden feature index to analyze & causally ablate (user override)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on ('cpu', 'cuda')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=["cifar10", "imagewoof", "imagenette", "imagenet"],
        help="Target dataset paradigm ('cifar10', 'imagewoof', 'imagenette', 'imagenet')",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data",
        help="Local file path directory for custom datasets",
    )
    parser.add_argument(
        "--eval_image_idx",
        type=int,
        default=None,
        help="Index of the image in the dataset to use for surgical evaluation (user override)",
    )
    parser.add_argument(
        "--context_patches",
        type=int,
        default=2,
        help="Number of patches of padding on each side of the central patch for context crops (e.g. 2 means a 5x5 crop)",
    )
    parser.add_argument(
        "--clip_activation_quantile",
        type=float,
        default=0.9,
        help="Activation quantile used to build activation-guided CLIP crops (0-1)",
    )
    parser.add_argument(
        "--clip_activation_min_tokens",
        type=int,
        default=4,
        help="Minimum number of active patch tokens kept in activation-guided CLIP crops",
    )
    parser.add_argument(
        "--clip_activation_max_tokens",
        type=int,
        default=9,
        help="Maximum activation-guided crop span (in patch tokens) along each image dimension",
    )
    parser.add_argument(
        "--min_grid_rel_drop",
        type=float,
        default=0.0,
        help="Minimum relative logit drop required for inclusion in rendered feature grids",
    )
    parser.add_argument(
        "--causal_eval_size",
        type=int,
        default=50,
        help="Number of images to evaluate causal metrics on (relative logit drop and accuracy drop)",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="mlp",
        choices=["mlp", "residual"],
        help="Submodule layer target type ('mlp' or 'residual')",
    )
    parser.add_argument(
        "--eval_split_ratio",
        type=float,
        default=0.3,
        help="Fraction of subset reserved for evaluation (generalization split)",
    )
    parser.add_argument(
        "--steering_factor",
        type=float,
        default=5.0,
        help="Multiplier factor for feature steering interventions",
    )
    parser.add_argument(
        "--min_baseline_abs",
        type=float,
        default=1e-2,
        help="Absolute value threshold for logging low confidence baseline logits",
    )
    parser.add_argument(
        "--export_human_eval",
        action="store_true",
        help="Export template CSV and grids for human-evaluation",
    )
    parser.add_argument(
        "--clip_top_k",
        type=int,
        default=3,
        help="Number of top CLIP concepts to store per feature",
    )
    parser.add_argument(
        "--clip_uncertainty_margin",
        type=float,
        default=0.01,
        help="If top-1 and top-2 CLIP scores differ by less than this margin, label as uncertain",
    )
    parser.add_argument(
        "--clip_uncertainty_margin_by_layer",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-layer CLIP uncertainty margins aligned with --layers order (falls back to --clip_uncertainty_margin)",
    )
    parser.add_argument(
        "--clip_max_dynamic_concepts",
        type=int,
        default=120,
        help="Maximum number of dynamic class-derived concepts to add to the fixed concept bank",
    )
    parser.add_argument(
        "--clip_dynamic_alias_mode",
        type=str,
        choices=["primary", "primary_secondary"],
        default="primary",
        help="Alias extraction mode for dynamic concepts from class labels",
    )
    parser.add_argument(
        "--clip_include_mapped_model_labels",
        action="store_true",
        help="Include mapped model class labels in the dynamic concept bank (dataset labels are always included when available)",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples used to estimate confidence intervals for mean relative logit drop (0 disables)",
    )
    parser.add_argument(
        "--bootstrap_ci",
        type=float,
        default=95.0,
        help="Bootstrap confidence level in percent for causal metrics (e.g. 95)",
    )
    parser.add_argument(
        "--disable_dynamic_concepts",
        action="store_true",
        help="Disable dataset/model-driven concept expansion and use only the fixed concept bank",
    )
    return parser.parse_args()


def get_top_active_features(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    dataloader: DataLoader,
    layer_idx: int,
    num_features: int = 10,
    target_type: str = "mlp",
    device: str = "cpu",
) -> List[int]:
    """
    Identifies the top N most active features across the dataset based on mean activation.
    """
    sae.eval()
    submodule = model_wrapper.get_submodule(layer_idx, target_type)
    hook = ActivationHook(submodule)
    hook.register()

    total_activations = torch.zeros(sae.hidden_dim, device=device)
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            _ = model_wrapper.model(images)
            activation = hook.activation
            if activation is None:
                continue
            # Discard CLS token (index 0)
            patch_tokens = activation[:, 1:, :]
            f = sae.encode(patch_tokens)  # [batch, num_patches, hidden_dim]
            total_activations += f.sum(dim=(0, 1))
            total_tokens += patch_tokens.shape[0] * patch_tokens.shape[1]

    hook.remove()
    mean_activations = total_activations / (total_tokens + 1e-8)
    top_values, top_indices = torch.topk(mean_activations, k=num_features)
    return top_indices.cpu().tolist()


def _download_fastai_dataset(data_dir: str, name: str, archive: str) -> str:
    """Generic downloader for fast.ai ImageNet subsets (imagewoof, imagenette)."""
    import tarfile
    import urllib.request

    dataset_dir = os.path.join(data_dir, archive.replace(".tgz", ""))
    val_dir = os.path.join(dataset_dir, "val")

    if os.path.exists(val_dir):
        return val_dir

    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, archive)
    url = f"https://s3.amazonaws.com/fast-ai-imageclas/{archive}"

    try:
        logger.info(f"{name} not found at {val_dir}. Auto-downloading from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        logger.info(f"Extracting {name} dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        os.remove(tar_path)
        logger.info(f"{name} setup completed at {val_dir}.")
        return val_dir
    except Exception as e:
        logger.error(f"Failed to auto-download {name}: {e}")
        raise e


def check_and_download_imagewoof(data_dir: str) -> str:
    return _download_fastai_dataset(data_dir, "ImageWoof", "imagewoof2-320.tgz")


def check_and_download_imagenette(data_dir: str) -> str:
    return _download_fastai_dataset(data_dir, "ImageNette", "imagenette2-320.tgz")


class HFImageDataset(Dataset):
    """Wraps a HuggingFace `datasets` image dataset to be compatible with torchvision transforms."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item.get("label", 0)
        if self.transform:
            image = self.transform(image)
        return image, label


def _normalize_label_name(label: str) -> str:
    """Normalize class names for robust cross-dataset label matching."""
    label = label.lower().replace("_", " ")
    label = re.sub(r"[^a-z0-9\s]", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def _label_aliases(label: str) -> List[str]:
    """Generate normalized aliases from a potentially comma-separated class label."""
    aliases = set()
    parts = [label] + [p.strip() for p in label.split(",") if p.strip()]
    for part in parts:
        norm = _normalize_label_name(part)
        if norm:
            aliases.add(norm)
    return list(aliases)


def _extract_dataset_label_names(dataset) -> Optional[List[str]]:
    """Extract ordered class names from torchvision-style or HF-style datasets."""
    if hasattr(dataset, "classes") and dataset.classes is not None:
        return [str(c) for c in dataset.classes]

    hf_dataset = getattr(dataset, "dataset", None)
    if hf_dataset is not None and hasattr(hf_dataset, "features"):
        label_feature = hf_dataset.features.get("label")
        if label_feature is not None:
            if hasattr(label_feature, "names") and label_feature.names is not None:
                return [str(x) for x in label_feature.names]
            if hasattr(label_feature, "num_classes") and hasattr(
                label_feature, "int2str"
            ):
                return [
                    str(label_feature.int2str(i)) for i in range(label_feature.num_classes)
                ]

    return None


def _extract_model_id2label(model_wrapper: ViTModelWrapper) -> Dict[int, str]:
    """Return model id2label map with integer ids."""
    model_config = getattr(model_wrapper.model, "config", None)
    if model_config is None:
        return {}

    id2label = getattr(model_config, "id2label", None)
    if not id2label:
        return {}

    normalized = {}
    for key, value in id2label.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        normalized[idx] = str(value)
    return normalized


def _build_dataset_to_model_label_map(
    dataset_labels: List[str], model_id2label: Dict[int, str]
) -> Dict[int, int]:
    """
    Build a mapping from dataset label indices to model logit indices via class names.
    """
    alias_to_model_idx: Dict[str, int] = {}
    for model_idx, model_label in model_id2label.items():
        for alias in _label_aliases(model_label):
            if alias and alias not in alias_to_model_idx:
                alias_to_model_idx[alias] = model_idx

    mapping: Dict[int, int] = {}
    for dataset_idx, dataset_label in enumerate(dataset_labels):
        match_idx = None
        for alias in _label_aliases(dataset_label):
            if alias in alias_to_model_idx:
                match_idx = alias_to_model_idx[alias]
                break
        if match_idx is not None:
            mapping[dataset_idx] = match_idx

    return mapping


def _dedupe_concepts(concepts: List[str]) -> List[str]:
    """Deduplicate concepts while preserving order, using normalized keys."""
    seen = set()
    deduped = []
    for concept in concepts:
        text = str(concept).strip()
        if not text:
            continue
        key = _normalize_label_name(text)
        if key and key not in seen:
            seen.add(key)
            deduped.append(text)
    return deduped


def _extract_label_aliases(label: str, alias_mode: str = "primary") -> List[str]:
    """Extract aliases from comma-separated class names with configurable granularity."""
    parts = [p.strip() for p in str(label).split(",") if p.strip()]
    if not parts:
        return []

    if alias_mode == "primary":
        return [parts[0]]

    if len(parts) == 1:
        return [parts[0]]
    return [parts[0], parts[1]]


def _filter_dynamic_concepts(dynamic_concepts: List[str]) -> List[str]:
    """Apply lightweight quality filters to reduce noisy dynamic concept aliases."""
    filtered = []
    for concept in _dedupe_concepts(dynamic_concepts):
        norm = _normalize_label_name(concept)
        if len(norm) < 3:
            continue
        if len(norm.split()) > 4:
            continue
        filtered.append(concept)
    return filtered


def _build_candidate_concepts(
    base_concepts: List[str],
    dataset_label_names: Optional[List[str]],
    mapped_model_labels: Optional[List[str]],
    max_dynamic_concepts: int,
    alias_mode: str = "primary",
    include_model_label_concepts: bool = False,
) -> tuple[List[str], List[str]]:
    """
    Build a hybrid concept bank:
      - fixed base concepts for comparability,
      - dynamic dataset/model concepts for better coverage.
    """
    dynamic = []
    if dataset_label_names:
        for label in dataset_label_names:
            dynamic.extend(_extract_label_aliases(label, alias_mode=alias_mode))
    if include_model_label_concepts and mapped_model_labels:
        for label in mapped_model_labels:
            dynamic.extend(_extract_label_aliases(label, alias_mode=alias_mode))

    dynamic = _filter_dynamic_concepts(dynamic)
    if max_dynamic_concepts > 0:
        dynamic = dynamic[:max_dynamic_concepts]
    else:
        dynamic = []

    merged = _dedupe_concepts(list(base_concepts) + dynamic)
    return merged, dynamic


def _resolve_layer_context_patches(
    base_context_patches: int, layer_idx: int, grid_size: int
) -> int:
    """
    Use larger context in lower/mid layers where local patches are less semantically stable.
    """
    extra = 0
    if layer_idx <= 3:
        extra = 2
    elif layer_idx <= 7:
        extra = 1

    resolved = base_context_patches + extra
    max_allowed = min(4, max(0, grid_size // 2))
    return max(0, min(int(resolved), max_allowed))


def _resolve_layer_uncertainty_margin(
    default_margin: float,
    layer_position: int,
    per_layer_margins: Optional[List[float]],
) -> float:
    """Resolve CLIP uncertainty margin with optional per-layer overrides."""
    if not per_layer_margins:
        return float(default_margin)
    if layer_position < len(per_layer_margins):
        return float(per_layer_margins[layer_position])
    return float(per_layer_margins[-1])


def _bootstrap_mean_ci(
    values: List[float],
    bootstrap_samples: int,
    ci_level: float,
    seed: int,
) -> Tuple[float, float]:
    """Estimate confidence interval for a mean via non-parametric bootstrap."""
    if bootstrap_samples <= 0 or not values:
        return float("nan"), float("nan")

    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(bootstrap_samples, arr.size), replace=True)
    means = draws.mean(axis=1)
    alpha = (100.0 - ci_level) / 2.0
    lower = float(np.percentile(means, alpha))
    upper = float(np.percentile(means, 100.0 - alpha))
    return lower, upper


def main():
    args = parse_args()

    # 1. Reproducibility & Device Configuration
    set_seed(42)
    device = args.device
    logger.info(f"Running pipeline on target device: {device}")

    if args.subset_size < 200:
        logger.warning(
            "WARNING: The subset_size is set to less than 200. Results will be unreliable."
        )

    if not 0.0 <= args.clip_activation_quantile <= 1.0:
        raise ValueError("--clip_activation_quantile must be in [0, 1].")
    if args.clip_activation_min_tokens < 1:
        raise ValueError("--clip_activation_min_tokens must be >= 1.")
    if args.clip_activation_max_tokens < 1:
        raise ValueError("--clip_activation_max_tokens must be >= 1.")
    if args.clip_uncertainty_margin < 0.0:
        raise ValueError("--clip_uncertainty_margin must be >= 0.")
    if args.clip_uncertainty_margin_by_layer is not None:
        if any(m < 0.0 for m in args.clip_uncertainty_margin_by_layer):
            raise ValueError("All --clip_uncertainty_margin_by_layer values must be >= 0.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap_samples must be >= 0.")
    if not 0.0 < args.bootstrap_ci < 100.0:
        raise ValueError("--bootstrap_ci must be between 0 and 100.")

    # 2. Loading Vision Transformer Backbone
    model_wrapper = ViTModelWrapper(model_name=args.model, device=device)

    # Output directory for all generated artifacts
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    # 3. Data Loader setup
    # Derive normalization constants and target size from model_wrapper.processor (Task 11)
    processor = getattr(model_wrapper, "processor", None)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    target_size = 224

    if processor is not None:
        try:
            p_mean = getattr(processor, "image_mean", None)
            p_std = getattr(processor, "image_std", None)
            if p_mean is not None:
                mean = p_mean
            if p_std is not None:
                std = p_std

            p_size = getattr(processor, "size", None)
            if p_size is not None:
                if isinstance(p_size, dict):
                    if "height" in p_size:
                        target_size = p_size["height"]
                    elif "shortest_edge" in p_size:
                        target_size = p_size["shortest_edge"]
                elif isinstance(p_size, int):
                    target_size = p_size
                elif isinstance(p_size, (list, tuple)):
                    target_size = p_size[0]
        except Exception as e:
            logger.warning(
                f"Error extracting normalization/size stats from processor: {e}. Using defaults."
            )

    logger.info(
        f"Derived image size: {target_size}, normalization mean: {mean}, std: {std}"
    )

    # Lanczos preserves sharp edges when upscaling low-res datasets (e.g. CIFAR 32x32 → 224x224)
    cifar_transform = transforms.Compose(
        [
            transforms.Resize(
                (target_size, target_size), interpolation=InterpolationMode.LANCZOS
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    resize_size = int(target_size * 256 / 224) if target_size else 256
    imagenet_style_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    if args.dataset == "cifar10":
        logger.info("Setting up CIFAR-10 test dataset (Lanczos upscaling)...")
        dataset = CIFAR10(
            root=args.dataset_path,
            train=False,
            download=True,
            transform=cifar_transform,
        )
    elif args.dataset == "imagewoof":
        try:
            val_dir = check_and_download_imagewoof(args.dataset_path)
            logger.info(f"Setting up ImageWoof dataset from {val_dir}...")
            from torchvision.datasets import ImageFolder

            dataset = ImageFolder(root=val_dir, transform=imagenet_style_transform)
        except Exception as e:
            logger.warning(f"Could not load ImageWoof: {e}. Falling back to CIFAR-10.")
            dataset = CIFAR10(
                root="./data", train=False, download=True, transform=cifar_transform
            )
    elif args.dataset == "imagenette":
        try:
            val_dir = check_and_download_imagenette(args.dataset_path)
            logger.info(
                f"Setting up ImageNette (10-class ImageNet subset) from {val_dir}..."
            )
            from torchvision.datasets import ImageFolder

            dataset = ImageFolder(root=val_dir, transform=imagenet_style_transform)
        except Exception as e:
            logger.warning(f"Could not load ImageNette: {e}. Falling back to CIFAR-10.")
            dataset = CIFAR10(
                root="./data", train=False, download=True, transform=cifar_transform
            )
    elif args.dataset == "imagenet":
        # ImageNet-100: try local path first, then HuggingFace datasets
        target_path = args.dataset_path
        if target_path == "./data":
            target_path = os.path.join(args.dataset_path, "imagenet-100", "val")

        from torchvision.datasets import ImageFolder

        if os.path.exists(target_path):
            logger.info(f"Setting up ImageNet-100 from local path {target_path}...")
            dataset = ImageFolder(root=target_path, transform=imagenet_style_transform)
        else:
            try:
                from datasets import load_dataset

                logger.info(
                    "Local ImageNet-100 not found. Downloading via HuggingFace datasets..."
                )
                hf_ds = load_dataset("clane9/imagenet-100", split="validation")
                dataset = HFImageDataset(hf_ds, transform=imagenet_style_transform)
                logger.info(
                    f"ImageNet-100 loaded from HuggingFace ({len(dataset)} images)."
                )
            except Exception as e:
                logger.warning(
                    f"Could not load ImageNet-100: {e}. Falling back to CIFAR-10. "
                    "To use ImageNet-100, either provide a local path via --dataset_path "
                    "or install the 'datasets' library (pip install datasets)."
                )
                dataset = CIFAR10(
                    root="./data", train=False, download=True, transform=cifar_transform
                )
    else:
        target_path = args.dataset_path
        if target_path == "./data":
            target_path = os.path.join(args.dataset_path, args.dataset)

        logger.info(f"Setting up {args.dataset} dataset from path {target_path}...")
        from torchvision.datasets import ImageFolder

        if os.path.exists(target_path):
            try:
                dataset = ImageFolder(
                    root=target_path, transform=imagenet_style_transform
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load ImageFolder at {target_path}: {e}. Falling back to CIFAR-10."
                )
                dataset = CIFAR10(
                    root="./data", train=False, download=True, transform=cifar_transform
                )
        else:
            logger.warning(
                f"Dataset path {target_path} not found. Falling back to CIFAR-10."
            )
            dataset = CIFAR10(
                root="./data", train=False, download=True, transform=cifar_transform
            )

    subset = torch.utils.data.Subset(
        dataset, range(min(args.subset_size, len(dataset)))
    )

    # Train/Eval Split (Task 5)
    eval_size = int(len(subset) * args.eval_split_ratio)
    train_size = len(subset) - eval_size
    if eval_size < 1:
        eval_size = 1
        train_size = max(0, len(subset) - eval_size)
    if train_size < 1:
        train_size = 1
        eval_size = max(0, len(subset) - train_size)

    train_subset, eval_subset = torch.utils.data.random_split(
        subset, [train_size, eval_size], generator=torch.Generator().manual_seed(42)
    )

    logger.info(
        f"Split dataset subset: {len(train_subset)} train samples, {len(eval_subset)} evaluation samples"
    )

    if len(eval_subset) < args.causal_eval_size:
        logger.warning(
            f"Evaluation subset size ({len(eval_subset)}) is smaller than requested "
            f"causal evaluation size ({args.causal_eval_size}). Causal evaluation will "
            f"be limited to {len(eval_subset)} images."
        )
    if len(eval_subset) < 5:
        logger.warning(
            f"Evaluation subset size ({len(eval_subset)}) is smaller than requested "
            f"number of exemplars (5). Some features may not have enough exemplars."
        )

    # Build DataLoaders (with comments guarding against mix-ups)
    # train_dataloader: fitting/selection
    train_dataloader = DataLoader(train_subset, batch_size=8, shuffle=False)
    # eval_dataloader: generalization eval
    eval_dataloader = DataLoader(eval_subset, batch_size=8, shuffle=False)

    # Train linear probe if using DINOv2 backbone (Task 4)
    if model_wrapper.model_type == "dinov2":
        logger.info(
            "DINOv2 has no native classifier head; a linear probe was trained on frozen embeddings for causal evaluation purposes."
        )
        # train_dataloader: fitting/selection
        model_wrapper.train_linear_probe(train_dataloader)

    dataset_num_classes = get_num_classes(dataset)
    logits_num_classes = None
    dataset_label_names = _extract_dataset_label_names(dataset)
    model_id2label = (
        _extract_model_id2label(model_wrapper) if model_wrapper.model_type == "vit" else {}
    )
    if model_wrapper.model_type == "dinov2" and model_wrapper.probe is not None:
        logits_num_classes = int(model_wrapper.probe.out_features)
    else:
        model_config = getattr(model_wrapper.model, "config", None)
        if model_config is not None:
            num_labels = getattr(model_config, "num_labels", None)
            if num_labels is not None:
                logits_num_classes = int(num_labels)

    labels_aligned = (
        logits_num_classes is not None and dataset_num_classes == logits_num_classes
    )

    dataset_to_model_label_map: Optional[Dict[int, int]] = None
    mapped_model_labels: List[str] = []
    if (
        not labels_aligned
        and logits_num_classes is not None
        and model_wrapper.model_type == "vit"
    ):
        if dataset_label_names and model_id2label:
            candidate_map = _build_dataset_to_model_label_map(
                dataset_labels=dataset_label_names,
                model_id2label=model_id2label,
            )
            coverage = len(candidate_map) / len(dataset_label_names)
            if coverage == 1.0:
                dataset_to_model_label_map = candidate_map
                labels_aligned = True
                mapped_model_labels = [
                    model_id2label[idx]
                    for idx in sorted(set(candidate_map.values()))
                    if idx in model_id2label
                ]
                logger.info(
                    "Resolved complete class-name mapping between dataset labels and model logits "
                    f"({len(candidate_map)} classes mapped)."
                )
            else:
                logger.warning(
                    "Class-name mapping between dataset and model is incomplete "
                    f"({len(candidate_map)}/{len(dataset_label_names)} mapped). "
                    "Accuracy-drop metrics will remain N/A."
                )

    if not labels_aligned:
        logger.warning(
            "Dataset/model label spaces are not aligned "
            f"(dataset classes={dataset_num_classes}, model logits={logits_num_classes}). "
            "Accuracy-drop metrics will be reported as N/A."
        )

    # 4. Multi-Layer Comparative Loop
    layers_to_compare = args.layers
    layer_comparison_summary = []
    discovered_features_detail = []
    layer_saes = {}
    layer_top_features = {}
    layer_histories = {}
    best_last_layer_feature_idx = None
    best_last_layer_drop = -float("inf")
    last_layer = layers_to_compare[-1]
    exemplars_cache = {}
    rendered_features = {}

    # Pre-load CLIP Auto-Labeler once to avoid reloading for every feature/layer
    labeler = CLIPAutoLabeler(device=device)
    dataset_concepts = {
        "imagewoof": [
            "fur",
            "eye",
            "nose",
            "ear",
            "tongue",
            "snout",
            "paw",
            "tail",
            "collar",
            "spotted pattern",
            "grass",
            "collie",
            "labrador",
        ],
        "imagenette": [
            "fish",
            "dog",
            "car",
            "church",
            "cassette player",
            "chain saw",
            "golf ball",
            "parachute",
            "gas pump",
            "horn",
            "sky",
            "grass",
            "water",
            "metal texture",
            "wood",
            "wheel",
            "fur",
            "eye",
            "scale pattern",
            "red color",
        ],
        "imagenet": [
            "animal",
            "dog",
            "bird",
            "fish",
            "insect",
            "building",
            "wasp",
            "plant",
            "furniture",
            "fur texture",
            "feather texture",
            "honeycomb pattern",
            "striped pattern",
            "spotted pattern",
            "geometric pattern",
            "pineapple",
            "eye",
            "scale pattern",
        ],
        "cifar10": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
            "sky",
            "water",
            "grass",
            "road",
            "wheel",
            "wing",
            "fur",
            "eye",
            "metal texture",
            "stripe",
        ],
    }
    base_concepts = dataset_concepts.get(
        args.dataset,
        dataset_concepts["cifar10"],
    )
    if args.disable_dynamic_concepts:
        candidate_concepts = _dedupe_concepts(base_concepts)
        dynamic_concepts = []
    else:
        candidate_concepts, dynamic_concepts = _build_candidate_concepts(
            base_concepts=base_concepts,
            dataset_label_names=dataset_label_names,
            mapped_model_labels=mapped_model_labels,
            max_dynamic_concepts=args.clip_max_dynamic_concepts,
            alias_mode=args.clip_dynamic_alias_mode,
            include_model_label_concepts=args.clip_include_mapped_model_labels,
        )
    logger.info(
        "CLIP concept bank configured with "
        f"{len(candidate_concepts)} concepts (base={len(base_concepts)}, dynamic={len(dynamic_concepts)}, "
        f"alias_mode={args.clip_dynamic_alias_mode}, include_model_labels={args.clip_include_mapped_model_labels})."
    )

    for i, layer in enumerate(layers_to_compare):
        layer_name = f"Layer {layer + 1}"
        logger.info("\n" + "=" * 50 + f"\nPROCESSING LAYER: {layer_name}\n" + "=" * 50)

        layer_context_patches = _resolve_layer_context_patches(
            base_context_patches=args.context_patches,
            layer_idx=layer,
            grid_size=model_wrapper.grid_size,
        )
        logger.info(
            f"Using context_patches={layer_context_patches} for {layer_name} "
            f"(base={args.context_patches})."
        )

        layer_uncertainty_margin = _resolve_layer_uncertainty_margin(
            default_margin=args.clip_uncertainty_margin,
            layer_position=i,
            per_layer_margins=args.clip_uncertainty_margin_by_layer,
        )
        logger.info(
            f"Using CLIP uncertainty margin={layer_uncertainty_margin:.4f} for {layer_name}."
        )

        # Determine L1 coefficient based on position in layers_to_compare (index i)
        if i < len(args.l1_coeff):
            l1_val = args.l1_coeff[i]
        else:
            l1_val = args.l1_coeff[-1]

        logger.info(f"Using L1 coefficient {l1_val:.2e} for {layer_name}")

        layer_grid_features_dict = {}

        # train_dataloader: fitting/selection
        activation_buffer = TokenActivationBuffer(
            model_wrapper=model_wrapper,
            dataloader=train_dataloader,
            layer_idx=layer,
            target_type=args.target_type,
            buffer_size=8192,
            sae_batch_size=256,
            device=device,
        )
        sae = SparseAutoencoder(
            d_model=model_wrapper.d_model, expansion_factor=args.expansion, tied=False
        ).to(device)

        # Train SAE
        epochs_val = args.epochs[i] if i < len(args.epochs) else args.epochs[-1]
        logger.info(f"Using {epochs_val} epochs for {layer_name}")

        history = train_sae(
            sae=sae,
            activation_buffer=activation_buffer,
            epochs=epochs_val,
            l1_coeff=l1_val,
            lr=1e-3,
            device=device,
        )

        final_r2 = history[-1]["r2"] if history else 0.0
        final_l0 = history[-1]["l0"] if history else 0.0
        layer_histories[layer_name] = history

        # top 10 most active features for this layer
        # train_dataloader: fitting/selection
        top_features = get_top_active_features(
            model_wrapper=model_wrapper,
            sae=sae,
            dataloader=train_dataloader,
            layer_idx=layer,
            num_features=10,
            target_type=args.target_type,
            device=device,
        )

        logger.info(
            f"Top 10 active features identified for {layer_name}: {top_features}"
        )

        layer_saes[layer] = sae
        layer_top_features[layer] = top_features

        layer_drops = []
        layer_acc_drops = []
        grid_candidates = []
        layer_rld_samples = []

        for idx, f_idx in enumerate(top_features):
            # eval_dataloader: generalization eval
            exemplars = get_top_activating_patches(
                model_wrapper=model_wrapper,
                sae=sae,
                dataloader=eval_dataloader,
                layer_idx=layer,
                feature_idx=f_idx,
                k=5,
                target_type=args.target_type,
                device=device,
            )
            # Store exemplars in cache for post-loop representative selection (Task 1 / 3)
            exemplars_cache[(layer, f_idx)] = exemplars

            # Auto-label with CLIP (passing derived mean/std)
            best_concept, best_score, all_scores, label_meta = labeler.label_feature(
                exemplars,
                candidate_concepts,
                patch_size=model_wrapper.patch_size,
                grid_size=model_wrapper.grid_size,
                context_patches=layer_context_patches,
                mean=np.array(mean),
                std=np.array(std),
                top_k=args.clip_top_k,
                uncertainty_margin=layer_uncertainty_margin,
                activation_crop_quantile=args.clip_activation_quantile,
                activation_crop_min_tokens=args.clip_activation_min_tokens,
                activation_crop_max_tokens=args.clip_activation_max_tokens,
            )

            top_concepts = label_meta.get("top_concepts", [])
            top_concepts_str = " | ".join(
                [f"{c}:{s:.4f}" for c, s in top_concepts]
            )
            best_raw_concept = label_meta.get("best_raw_concept", best_concept)
            is_uncertain = bool(label_meta.get("is_uncertain", False))
            clip_margin = float(label_meta.get("margin_top1_top2", 0.0))

            # Multi-image Causal Dataset Evaluation
            # eval_dataloader: generalization eval
            causal_res = evaluate_causal_dataset(
                model_wrapper=model_wrapper,
                sae=sae,
                dataloader=eval_dataloader,
                layer_idx=layer,
                feature_idx=f_idx,
                target_type=args.target_type,
                max_images=args.causal_eval_size,
                steering_factor=args.steering_factor,
                min_baseline_abs=args.min_baseline_abs,
                labels_aligned=labels_aligned,
                dataset_to_model_label_map=dataset_to_model_label_map,
                device=device,
            )

            rel_drop = causal_res["mean_rld"]
            steer_increase = causal_res["mean_sli"]
            accuracy_drop = causal_res["accuracy_drop"]
            feature_rld_samples = causal_res.get("relative_drops", [])
            layer_rld_samples.extend(feature_rld_samples)

            ci_low, ci_high = _bootstrap_mean_ci(
                values=feature_rld_samples,
                bootstrap_samples=args.bootstrap_samples,
                ci_level=args.bootstrap_ci,
                seed=42 + (layer * 1000) + idx,
            )

            layer_drops.append(rel_drop)
            if accuracy_drop is not None:
                layer_acc_drops.append(accuracy_drop)

            if layer == last_layer:
                if rel_drop > best_last_layer_drop:
                    best_last_layer_drop = rel_drop
                    best_last_layer_feature_idx = f_idx

            feat_result = {
                "Feature Index": int(f_idx),
                "Target Layer": layer_name,
                "Assigned CLIP Concept": best_concept,
                "Best Raw CLIP Concept": best_raw_concept,
                "CLIP Confidence Score": float(best_score),
                "CLIP Top Concepts": top_concepts_str,
                "CLIP Margin Top1-Top2": clip_margin,
                "CLIP Uncertain": is_uncertain,
                "CLIP Uncertainty Margin Used": float(layer_uncertainty_margin),
                "Context Patches Used": int(layer_context_patches),
                "Mean Relative Logit Drop (%)": float(rel_drop),
                "Mean Relative Logit Drop CI Low (%)": (
                    float(ci_low) if not np.isnan(ci_low) else None
                ),
                "Mean Relative Logit Drop CI High (%)": (
                    float(ci_high) if not np.isnan(ci_high) else None
                ),
                "Mean Steered Logit Increase (%)": float(steer_increase),
                "Accuracy Drop (%)": accuracy_drop,
                "Low Confidence Baseline Fraction": float(
                    causal_res["low_confidence_fraction"]
                ),
            }
            discovered_features_detail.append(feat_result)

            grid_candidates.append(
                {
                    "feat_idx": f_idx,
                    "exemplars": exemplars,
                    "concept": (
                        f"uncertain ({best_raw_concept})"
                        if is_uncertain
                        else best_concept
                    ),
                    "clip_score": best_score,
                    "rel_drop": rel_drop,
                }
            )

        # Select the 5 candidates with strongest causal effect, using CLIP score as tie-breaker.
        grid_candidates.sort(
            key=lambda x: (x["rel_drop"], x["clip_score"]), reverse=True
        )

        filtered_grid_candidates = [
            cand for cand in grid_candidates if cand["rel_drop"] >= args.min_grid_rel_drop
        ]
        if not filtered_grid_candidates:
            filtered_grid_candidates = grid_candidates

        for cand in filtered_grid_candidates[:5]:
            layer_grid_features_dict[cand["feat_idx"]] = {
                "exemplars": cand["exemplars"],
                "concept": cand["concept"],
            }
            grid_filename = f"feature_grid_layer{layer + 1}_feat{cand['feat_idx']}.png"
            rendered_features[(layer_name, cand["feat_idx"])] = grid_filename

        mean_logit_drop = sum(layer_drops) / (len(layer_drops) + 1e-8)
        layer_ci_low, layer_ci_high = _bootstrap_mean_ci(
            values=layer_rld_samples,
            bootstrap_samples=args.bootstrap_samples,
            ci_level=args.bootstrap_ci,
            seed=4242 + layer,
        )
        if np.isnan(layer_ci_low):
            layer_ci_str = "N/A"
        else:
            layer_ci_str = f"[{layer_ci_low:.4f}, {layer_ci_high:.4f}]%"

        if layer_acc_drops:
            mean_acc_drop = sum(layer_acc_drops) / (len(layer_acc_drops) + 1e-8)
            mean_acc_drop_str = f"{mean_acc_drop:.4f}%"
        else:
            mean_acc_drop_str = "N/A"

        layer_comparison_summary.append(
            {
                "Layer": layer_name,
                "R^2 Score": f"{final_r2:.4f}",
                "L_0 Norm": f"{final_l0:.2f}",
                "Mean Logit Drop": f"{mean_logit_drop:.4f}%",
                "Mean Logit Drop CI": layer_ci_str,
                "Mean Accuracy Drop": mean_acc_drop_str,
            }
        )

        # Save unified feature grid visualization for the current layer
        if layer_grid_features_dict:
            grid_path = os.path.join(
                out_dir, f"multi_feature_exemplar_grid_layer{layer + 1}.png"
            )
            save_feature_grid_visualization(
                model_wrapper=model_wrapper,
                sae=sae,
                top_features_dict=layer_grid_features_dict,
                output_path=grid_path,
                layer_idx=layer,
                patch_size=model_wrapper.patch_size,
                grid_size=model_wrapper.grid_size,
                device=device,
                context_patches=layer_context_patches,
            )

            # Save individual feature grid visualizations for each feature
            for feat_idx, feat_data in layer_grid_features_dict.items():
                single_feat_path = os.path.join(
                    out_dir, f"feature_grid_layer{layer + 1}_feat{feat_idx}.png"
                )
                save_feature_grid_visualization(
                    model_wrapper=model_wrapper,
                    sae=sae,
                    top_features_dict={feat_idx: feat_data},
                    output_path=single_feat_path,
                    layer_idx=layer,
                    patch_size=model_wrapper.patch_size,
                    grid_size=model_wrapper.grid_size,
                    device=device,
                    context_patches=layer_context_patches,
                )

    # 5. summary table to stdout and file
    markdown_table = (
        "\n"
        + "=" * 78
        + "\n"
        + "             MULTI-LAYER COMPARATIVE SUMMARY\n"
        + "=" * 78
        + "\n"
        + f"| {'Layer':10} | {'R^2 Score':9} | {'L_0 Norm':8} | {'Mean Logit Drop':17} | {'Mean Logit CI':23} | {'Mean Acc Drop':15} |\n"
        + "|"
        + "-" * 12
        + "|"
        + "-" * 11
        + "|"
        + "-" * 10
        + "|"
        + "-" * 19
        + "|"
        + "-" * 25
        + "|"
        + "-" * 17
        + "|\n"
    )
    for row in layer_comparison_summary:
        markdown_table += f"| {row['Layer']:10} | {row['R^2 Score']:9} | {row['L_0 Norm']:8} | {row['Mean Logit Drop']:17} | {row['Mean Logit Drop CI']:23} | {row['Mean Accuracy Drop']:15} |\n"
    markdown_table += "=" * 78 + "\n"

    print(markdown_table)

    summary_path = os.path.join(out_dir, "layer_comparison_summary.md")
    with open(summary_path, mode="w", encoding="utf-8") as f:
        f.write(markdown_table)
    logger.info(f"Saved layer comparison table to {summary_path}")

    # 7. Quantitative CSV Exporter
    csv_file_path = os.path.join(out_dir, "discovered_features_summary.csv")
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Feature Index",
                "Target Layer",
                "Assigned CLIP Concept",
                "Best Raw CLIP Concept",
                "CLIP Confidence Score",
                "CLIP Top Concepts",
                "CLIP Margin Top1-Top2",
                "CLIP Uncertain",
                "CLIP Uncertainty Margin Used",
                "Context Patches Used",
                "Mean Relative Logit Drop (%)",
                "Mean Relative Logit Drop CI Low (%)",
                "Mean Relative Logit Drop CI High (%)",
                "Mean Steered Logit Increase (%)",
                "Accuracy Drop (%)",
                "Low Confidence Baseline Fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(discovered_features_detail)
    logger.info(f"Saved quantitative summary to {csv_file_path}")

    # 8. Post-loop Representative Feature Selection (Task 2)
    last_layer_sae = layer_saes[last_layer]
    last_layer_features = layer_top_features[last_layer]

    representative_feat = (
        args.feature_idx
        if args.feature_idx is not None
        else (
            best_last_layer_feature_idx
            if best_last_layer_feature_idx is not None
            else (last_layer_features[0] if last_layer_features else 100)
        )
    )
    if args.feature_idx is not None:
        logger.info(
            f"User-specified Feature Index override: plotting Feature {representative_feat} instead of best feature."
        )
    else:
        logger.info(
            f"Selected Feature {representative_feat} for Dose-Response curve (ablation drop: {best_last_layer_drop:.4f}%)"
        )

    # 8.5 Post-loop Evaluation Image Selection (Task 3)
    first_img_tensor = None
    eval_image = None
    first_img_label = 0

    if args.eval_image_idx is not None:
        eval_idx = args.eval_image_idx
        # A manually-supplied eval_image_idx continues to index the full dataset (not eval_subset)
        if eval_idx < 0 or eval_idx >= len(dataset):
            logger.warning(
                f"Evaluation image index {eval_idx} is out of bounds for dataset of size {len(dataset)}. Defaulting to index 0."
            )
            eval_idx = 0
        first_img_tensor, first_img_label = dataset[eval_idx]
        eval_image = first_img_tensor.unsqueeze(0).to(device)
        logger.info(
            f"Using manually requested evaluation image at index {eval_idx} from the full dataset (label: {first_img_label})"
        )
    else:
        logger.info(
            f"No evaluation image index specified. Checking cache/top exemplar for Feature {representative_feat}..."
        )
        cached_exemplars = exemplars_cache.get((last_layer, representative_feat))
        if cached_exemplars:
            first_img_tensor = cached_exemplars[0]["full_image"]
            eval_image = first_img_tensor.unsqueeze(0).to(device)
            first_img_label = cached_exemplars[0].get("label", 0)
            logger.info(
                f"Successfully reused cached top exemplar image for Feature {representative_feat} (activation: {cached_exemplars[0]['activation']:.4f}, label: {first_img_label})"
            )
        else:
            # eval_dataloader: generalization eval
            exemplars = get_top_activating_patches(
                model_wrapper=model_wrapper,
                sae=last_layer_sae,
                dataloader=eval_dataloader,
                layer_idx=last_layer,
                feature_idx=representative_feat,
                k=1,
                target_type=args.target_type,
                device=device,
            )
            if exemplars:
                first_img_tensor = exemplars[0]["full_image"]
                eval_image = first_img_tensor.unsqueeze(0).to(device)
                first_img_label = exemplars[0].get("label", 0)
                logger.info(
                    f"Successfully selected top exemplar image for Feature {representative_feat} (activation: {exemplars[0]['activation']:.4f}, label: {first_img_label})"
                )
            else:
                logger.warning(
                    f"No activating exemplar found for Feature {representative_feat}. Falling back to dataset index 0."
                )
                first_img_tensor, first_img_label = dataset[0]
                eval_image = first_img_tensor.unsqueeze(0).to(device)

    # Class name resolution
    class_name = "unknown"
    if hasattr(dataset, "classes") and dataset.classes is not None:
        try:
            class_name = dataset.classes[first_img_label]
        except Exception:
            pass
    elif (
        hasattr(dataset, "dataset")
        and hasattr(dataset.dataset, "features")
        and "label" in dataset.dataset.features
    ):
        try:
            class_name = dataset.dataset.features["label"].int2str(first_img_label)
        except Exception:
            pass

    # Get baseline class prediction for eval_image
    with torch.no_grad():
        out = model_wrapper.model(eval_image)
        logits = model_wrapper.get_logits(out)
        predicted_class_idx = logits.argmax(dim=-1).item()

    logger.info(
        f"Selected evaluation image (Label index: {first_img_label}, Class: {class_name})"
    )
    logger.info(
        f"Chosen evaluation image predicted class index: {predicted_class_idx} (Label index: {first_img_label})"
    )

    # 9. Single-Image Causal Evaluation (Task 10)
    logger.info(
        "Evaluating relative logit drop on the single chosen evaluation image..."
    )
    evaluate_relative_logit_drop(
        model_wrapper=model_wrapper,
        sae=last_layer_sae,
        images=eval_image,
        layer_idx=last_layer,
        feature_idx=representative_feat,
        target_class_idx=predicted_class_idx,
        target_type=args.target_type,
    )

    # 10. Plot Dose-Response for representative feature
    plot_dose_response(
        model_wrapper=model_wrapper,
        sae=last_layer_sae,
        images=eval_image,
        layer_idx=last_layer,
        feature_idx=representative_feat,
        target_class_idx=predicted_class_idx,
        target_type=args.target_type,
        save_path=os.path.join(out_dir, "dose_response_curve.png"),
    )

    # 11. Plot SAE training convergence curves
    if layer_histories:
        plot_training_curves(
            layer_histories, save_path=os.path.join(out_dir, "sae_training_curves.png")
        )

    # 12. Generate spatial activation heatmap for representative feature
    save_feature_activation_heatmap(
        model_wrapper=model_wrapper,
        sae=last_layer_sae,
        image=first_img_tensor,
        layer_idx=last_layer,
        feature_idx=representative_feat,
        target_type=args.target_type,
        save_path=os.path.join(out_dir, "feature_activation_heatmap.png"),
        device=device,
    )

    # 13. Export human evaluation templates if enabled (Task 7)
    if args.export_human_eval:
        import shutil

        human_eval_dir = os.path.join(out_dir, "human_eval")
        os.makedirs(human_eval_dir, exist_ok=True)

        template_path = os.path.join(human_eval_dir, "human_eval_template.csv")
        with open(template_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Feature Index",
                    "Target Layer",
                    "Image Path",
                    "CLIP Concept",
                    "CLIP Confidence",
                    "CLIP Top Concepts",
                    "Rater1 Label",
                    "Rater2 Label",
                    "Rater1 Confidence (1-5)",
                    "Rater2 Confidence (1-5)",
                    "Notes",
                ]
            )

            for feat in discovered_features_detail:
                feat_idx = feat["Feature Index"]
                layer_name = feat["Target Layer"]

                # Only write template entries for features that were actually rendered
                if (layer_name, feat_idx) not in rendered_features:
                    continue

                grid_filename = rendered_features[(layer_name, feat_idx)]
                concept = feat["Assigned CLIP Concept"]
                score = f"{feat['CLIP Confidence Score']:.4f}"
                top_concepts = feat.get("CLIP Top Concepts", "")

                src_image_path = os.path.join(out_dir, grid_filename)
                dest_image_path = os.path.join(human_eval_dir, grid_filename)

                if os.path.exists(src_image_path):
                    shutil.copy(src_image_path, dest_image_path)

                writer.writerow(
                    [
                        feat_idx,
                        layer_name,
                        f"human_eval/{grid_filename}",
                        concept,
                        score,
                        top_concepts,
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
        logger.info(f"Exported human evaluation template and grids to {human_eval_dir}")

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
