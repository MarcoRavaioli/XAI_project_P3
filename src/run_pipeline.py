import os

os.environ["MPLBACKEND"] = "agg"
import csv
import argparse
import logging
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR10

from typing import List

from model_loader import set_seed, ViTModelWrapper, ActivationHook
from sae import SparseAutoencoder
from caching_and_training import TokenActivationBuffer, train_sae
from interpretability import (
    get_top_activating_patches,
    CLIPAutoLabeler,
    save_feature_grid_visualization,
    save_feature_activation_heatmap,
)
from causal_eval import (
    plot_dose_response,
    perform_causal_intervention,
    plot_training_curves,
)

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
        "--layer",
        type=int,
        default=5,
        help="0-indexed layers targeting block depth (5 = Layer 6, 10 = Layer 11)",
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
        default=50,
        help="Size of subset for rapid pipeline testing",
    )
    parser.add_argument(
        "--feature_idx",
        type=int,
        default=100,
        help="SAE hidden feature index to analyze & causally ablate",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on ('cpu', 'cuda')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
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
        default=0,
        help="Index of the image in the dataset to use for surgical evaluation",
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
    import urllib.request
    import tarfile

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


def main():
    args = parse_args()

    # 1. Reproducibility & Device Configuration
    set_seed(42)
    device = args.device
    logger.info(f"Running pipeline on target device: {device}")

    # 2. Loading Vision Transformer Backbone
    model_wrapper = ViTModelWrapper(model_name=args.model, device=device)

    # Output directory for all generated artifacts
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    # 3. Data Loader setup
    # Lanczos preserves sharp edges when upscaling low-res datasets (e.g. CIFAR 32x32 → 224x224)
    cifar_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    imagenet_style_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    dataloader = DataLoader(subset, batch_size=8, shuffle=False)

    # 4. Multi-Layer Comparative Loop
    layers_to_compare = [5, 10]  # Layer 6 and Layer 11
    layer_comparison_summary = []
    discovered_features_detail = []
    layer_saes = {}
    layer_top_features = {}
    layer_histories = {}
    best_layer11_feature_idx = None
    best_layer11_drop = -float("inf")

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
            "snake",
            "building",
            "wasp",
            "plant",
            "furniture",
            "person",
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
    candidate_concepts = dataset_concepts.get(
        args.dataset,
        dataset_concepts["cifar10"],
    )

    # Select evaluation image from dataset for surgical visual intervention evaluation
    eval_idx = args.eval_image_idx
    if eval_idx < 0 or eval_idx >= len(dataset):
        logger.warning(
            f"Evaluation image index {eval_idx} is out of bounds for dataset of size {len(dataset)}. "
            f"Defaulting to index 0."
        )
        eval_idx = 0

    first_img_tensor, first_img_label = dataset[eval_idx]
    eval_image = first_img_tensor.unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

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
        logits = out.logits if hasattr(out, "logits") else out.last_hidden_state[:, 0]
        predicted_class_idx = logits.argmax(dim=-1).item()

    logger.info(
        f"Selected evaluation image at index {eval_idx} (Label index: {first_img_label}, Class: {class_name})"
    )
    logger.info(
        f"Image {eval_idx} predicted class index: {predicted_class_idx} (Label index: {first_img_label})"
    )

    for i, layer in enumerate(layers_to_compare):
        layer_name = f"Layer {layer + 1}"
        logger.info(f"\n" + "=" * 50 + f"\nPROCESSING LAYER: {layer_name}\n" + "=" * 50)

        # Determine L1 coefficient based on position in layers_to_compare (index i)
        if i < len(args.l1_coeff):
            l1_val = args.l1_coeff[i]
        else:
            l1_val = args.l1_coeff[-1]

        logger.info(f"Using L1 coefficient {l1_val:.2e} for {layer_name}")

        layer_grid_features_dict = {}

        activation_buffer = TokenActivationBuffer(
            model_wrapper=model_wrapper,
            dataloader=dataloader,
            layer_idx=layer,
            target_type="mlp",
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
        top_features = get_top_active_features(
            model_wrapper=model_wrapper,
            sae=sae,
            dataloader=dataloader,
            layer_idx=layer,
            num_features=10,
            target_type="mlp",
            device=device,
        )

        logger.info(
            f"Top 10 active features identified for {layer_name}: {top_features}"
        )

        layer_saes[layer] = sae
        layer_top_features[layer] = top_features

        layer_drops = []

        for idx, f_idx in enumerate(top_features):
            exemplars = get_top_activating_patches(
                model_wrapper=model_wrapper,
                sae=sae,
                dataloader=dataloader,
                layer_idx=layer,
                feature_idx=f_idx,
                k=5,
                target_type="mlp",
                device=device,
            )

            # Auto-label with CLIP
            best_concept, best_score, all_scores = labeler.label_feature(
                exemplars,
                candidate_concepts,
                patch_size=model_wrapper.patch_size,
                grid_size=model_wrapper.grid_size,
            )

            # Baseline logit
            with torch.no_grad():
                out_baseline = model_wrapper.model(eval_image)
                logits_base = (
                    out_baseline.logits
                    if hasattr(out_baseline, "logits")
                    else out_baseline.last_hidden_state[:, 0]
                )
                baseline_logit = logits_base[:, predicted_class_idx].mean().item()

            # Ablation intervention
            logits_ablated = perform_causal_intervention(
                model_wrapper,
                sae,
                eval_image,
                layer_idx=layer,
                feature_idx=f_idx,
                intervention_type="ablation",
                scaling_factor=0.0,
                target_type="mlp",
            )
            ablated_logit = logits_ablated[:, predicted_class_idx].mean().item()
            rel_drop = (
                (baseline_logit - ablated_logit) / (abs(baseline_logit) + 1e-8) * 100
            )
            layer_drops.append(rel_drop)

            if layer == 10:
                if rel_drop > best_layer11_drop:
                    best_layer11_drop = rel_drop
                    best_layer11_feature_idx = f_idx

            # Steering intervention
            logits_steered = perform_causal_intervention(
                model_wrapper,
                sae,
                eval_image,
                layer_idx=layer,
                feature_idx=f_idx,
                intervention_type="steering",
                scaling_factor=5.0,
                target_type="mlp",
            )
            steered_logit = logits_steered[:, predicted_class_idx].mean().item()
            steer_increase = (
                (steered_logit - baseline_logit) / (abs(baseline_logit) + 1e-8) * 100
            )

            feat_result = {
                "Feature Index": f_idx,
                "Target Layer": layer_name,
                "Assigned CLIP Concept": best_concept,
                "CLIP Confidence Score": f"{best_score:.4f}",
                "Baseline Logit": f"{baseline_logit:.4f}",
                "Ablated Logit": f"{ablated_logit:.4f}",
                "Relative Logit Drop (%)": f"{rel_drop:.4f}",
                "Steered Logit Increase (%)": f"{steer_increase:.4f}",
            }
            discovered_features_detail.append(feat_result)

            if len(layer_grid_features_dict) < 5:
                layer_grid_features_dict[f_idx] = {
                    "exemplars": exemplars,
                    "concept": best_concept,
                }

        mean_logit_drop = sum(layer_drops) / (len(layer_drops) + 1e-8)

        layer_comparison_summary.append(
            {
                "Layer": layer_name,
                "R^2 Score": f"{final_r2:.4f}",
                "L_0 Norm": f"{final_l0:.2f}",
                "Mean Logit Drop": f"{mean_logit_drop:.4f}%",
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
            )
            if layer == 10:
                import shutil

                shutil.copy(
                    grid_path, os.path.join(out_dir, "multi_feature_exemplar_grid.png")
                )

    # 5. summary table to stdout and file
    markdown_table = (
        "\n"
        + "=" * 59
        + "\n"
        + "             MULTI-LAYER COMPARATIVE SUMMARY\n"
        + "=" * 59
        + "\n"
        + f"| {'Layer':10} | {'R^2 Score':9} | {'L_0 Norm':8} | {'Mean Logit Drop':17} |\n"
        + "|"
        + "-" * 12
        + "|"
        + "-" * 11
        + "|"
        + "-" * 10
        + "|"
        + "-" * 19
        + "|\n"
    )
    for row in layer_comparison_summary:
        markdown_table += f"| {row['Layer']:10} | {row['R^2 Score']:9} | {row['L_0 Norm']:8} | {row['Mean Logit Drop']:17} |\n"
    markdown_table += "=" * 59 + "\n"

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
                "CLIP Confidence Score",
                "Baseline Logit",
                "Ablated Logit",
                "Relative Logit Drop (%)",
                "Steered Logit Increase (%)",
            ],
        )
        writer.writeheader()
        writer.writerows(discovered_features_detail)
    logger.info(f"Saved quantitative summary to {csv_file_path}")

    # 8. Plot Dose-Response for a representative feature (Feature with largest causal drop on Layer 11)
    logger.info("Generating Causal Dose-Response curve for representation...")
    layer11_sae = layer_saes[10]
    layer11_features = layer_top_features[10]

    representative_feat = (
        best_layer11_feature_idx
        if best_layer11_feature_idx is not None
        else (layer11_features[0] if layer11_features else 100)
    )
    logger.info(
        f"Selected Feature {representative_feat} for Dose-Response curve (ablation drop: {best_layer11_drop:.4f}%)"
    )

    plot_dose_response(
        model_wrapper=model_wrapper,
        sae=layer11_sae,
        images=eval_image,
        layer_idx=10,
        feature_idx=representative_feat,
        target_class_idx=predicted_class_idx,
        target_type="mlp",
        save_path=os.path.join(out_dir, "dose_response_curve.png"),
    )

    # 9. Plot SAE training convergence curves
    if layer_histories:
        plot_training_curves(
            layer_histories, save_path=os.path.join(out_dir, "sae_training_curves.png")
        )

    # 10. Generate spatial activation heatmap for the most causally active Layer 11 feature
    if best_layer11_feature_idx is not None:
        save_feature_activation_heatmap(
            model_wrapper=model_wrapper,
            sae=layer11_sae,
            image=first_img_tensor,
            layer_idx=10,
            feature_idx=best_layer11_feature_idx,
            target_type="mlp",
            save_path=os.path.join(out_dir, "feature_activation_heatmap.png"),
            device=device,
        )

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
