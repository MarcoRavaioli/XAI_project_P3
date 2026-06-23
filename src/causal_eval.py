import os

os.environ["MPLBACKEND"] = "agg"
import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import torch
from torch.utils.data import DataLoader

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from model_loader import ActivationHook, ViTModelWrapper
from sae import SparseAutoencoder

logger = logging.getLogger(__name__)


def perform_causal_intervention(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    images: torch.Tensor,
    layer_idx: int,
    feature_idx: int,
    intervention_type: str = "ablation",
    ablation_strength: float = 0.0,
    steering_factor: float = 1.0,
    target_type: str = "mlp",
) -> torch.Tensor:
    """
    Forward pass with hook interventions on the target SAE feature.

    Citations:
        - Elhage et al. [2021] ("A Mathematical Framework for Transformer Circuits")
    """
    model_wrapper.model.eval()
    device = next(model_wrapper.model.parameters()).device
    images = images.to(device)

    submodule = model_wrapper.get_submodule(layer_idx, target_type)  # target submodule

    def intervention_callback(x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        cls_token = x[:, 0:1, :]  # keep gobal classification token untouched
        patch_tokens = x[:, 1:, :]  # [batch, num_patches, d_model]

        # now we project to SAE space to obtain hidden activations
        f = sae.encode(patch_tokens)  # [batch, num_patches, hidden_dim]
        f_j = f[:, :, feature_idx]  # Target feature activation: [batch, num_patches]

        # we then retrieve target feature's reconstruction column vector from decoder
        W_dec_j = sae.W_dec[:, feature_idx]  # [d_model]

        if intervention_type == "ablation":
            # Subtract only the visual feature projection from the residual stream
            # ablation_strength 0.0 is full ablation, 1.0 is baseline (no-op)
            ablation_term = 1.0 - ablation_strength
            patch_tokens_modified = patch_tokens - ablation_term * f_j.unsqueeze(
                -1
            ) * W_dec_j.view(1, 1, -1)

        elif intervention_type == "steering":
            # Scale up target feature's activation
            # steering_factor 1.0 is baseline, >1.0 is steering
            patch_tokens_modified = patch_tokens + (
                steering_factor - 1.0
            ) * f_j.unsqueeze(-1) * W_dec_j.view(1, 1, -1)

        else:
            raise ValueError(f"Unsupported intervention_type: {intervention_type}")

        x_modified = torch.cat([cls_token, patch_tokens_modified], dim=1)
        return x_modified

    hook = ActivationHook(submodule, callback=intervention_callback)
    hook.register()

    try:
        with torch.no_grad():
            outputs = model_wrapper.model(images)
            logits = model_wrapper.get_logits(outputs)
    finally:
        hook.remove()

    return logits


def evaluate_relative_logit_drop(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    images: torch.Tensor,
    layer_idx: int,
    feature_idx: int,
    target_class_idx: int,
    target_type: str = "mlp",
) -> Tuple[float, float, float]:
    """
    Computes the Relative Logit Drop metric for the target class under feature ablation.

    Relative Logit Drop = (Logit_baseline - Logit_ablated) / Logit_baseline

    Output:
        baseline_logit: Class prediction logit before intervention.
        ablated_logit: Class prediction logit after surgical ablation.
        relative_drop: Computed relative logit drop value.
    """
    # Baseline logit
    with torch.no_grad():
        outputs_baseline = model_wrapper.model(images.to(model_wrapper.device))
        logits_baseline = model_wrapper.get_logits(outputs_baseline)

        baseline_logit = logits_baseline[:, target_class_idx].mean().item()

    logits_ablated = perform_causal_intervention(
        model_wrapper,
        sae,
        images,
        layer_idx,
        feature_idx,
        intervention_type="ablation",
        ablation_strength=0.0,
        target_type=target_type,
    )
    ablated_logit = logits_ablated[:, target_class_idx].mean().item()

    # Relative drop
    relative_drop = (baseline_logit - ablated_logit) / (abs(baseline_logit) + 1e-8)

    logger.info(
        f"Causal Ablation on Feature {feature_idx} for Class {target_class_idx} | "
        f"Baseline Logit: {baseline_logit:.4f} | "
        f"Ablated Logit: {ablated_logit:.4f} | "
        f"Relative Logit Drop: {relative_drop * 100:.2f}%"
    )

    return baseline_logit, ablated_logit, relative_drop


def plot_dose_response(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    images: torch.Tensor,
    layer_idx: int,
    feature_idx: int,
    target_class_idx: int,
    target_type: str = "mlp",
    save_path: str = "dose_response.png",
):
    """
    Dose-response plot showing Relative Logit Drop vs Ablation Strength.
    Varies ablation strength (0.0 to 1.0) and plots the curves.
    """
    ablation_strengths = np.linspace(
        0.0, 1.0, 6
    )  # 0.0 = full ablation, 1.0 = baseline (0% ablation)
    logits = []

    # Baseline logit to compute drops
    with torch.no_grad():
        outputs_baseline = model_wrapper.model(images.to(model_wrapper.device))
        logits_baseline = model_wrapper.get_logits(outputs_baseline)
        baseline_logit = logits_baseline[:, target_class_idx].mean().item()

    for strength in ablation_strengths:
        logits_intervened = perform_causal_intervention(
            model_wrapper,
            sae,
            images,
            layer_idx,
            feature_idx,
            intervention_type="ablation",
            ablation_strength=strength,
            target_type=target_type,
        )
        avg_logit = logits_intervened[:, target_class_idx].mean().item()
        logits.append(avg_logit)

    # Relative logit drop = (baseline - ablated) / baseline
    # strength = 1.0 (baseline) -> drop should be 0.0
    # strength = 0.0 (full ablation) -> max drop
    relative_drops = [
        (baseline_logit - logit) / (abs(baseline_logit) + 1e-8) for logit in logits
    ]
    plt.figure(figsize=(7, 4.5))
    percent_ablation = (
        1.0 - ablation_strengths
    ) * 100  # so 100% ablation is at ablation_strength=0.0

    plt.plot(
        percent_ablation,
        [d * 100 for d in relative_drops],
        marker="o",
        linewidth=2,
        color="#e06666",
    )
    plt.title(
        f"Causal Dose-Response: SAE Feature {feature_idx}",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Ablation Strength (%)", fontsize=10)
    plt.ylabel("Relative Logit Drop (%)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Dose-response plot saved to {save_path}")


def plot_training_curves(
    all_histories: dict,
    save_path: str = "sae_training_curves.png",
):
    """
    Plots SAE training convergence curves for cross-layer comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [("loss", "Training Loss"), ("r2", "R² Score"), ("l0", "L₀ Sparsity")]
    colors = ["#e06666", "#6fa8dc", "#93c47d", "#f6b26b"]

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx]
        for i, (layer_name, history) in enumerate(all_histories.items()):
            epochs = [h["epoch"] for h in history]
            values = [h[key] for h in history]
            ax.plot(
                epochs,
                values,
                marker="o",
                linewidth=2,
                markersize=5,
                color=colors[i % len(colors)],
                label=layer_name,
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Training curves saved to {save_path}")


def evaluate_causal_dataset(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    dataloader: DataLoader,
    layer_idx: int,
    feature_idx: int,
    target_type: str = "mlp",
    max_images: int = 50,
    steering_factor: float = 5.0,
    min_baseline_abs: float = 1e-2,
    labels_aligned: bool = True,
    dataset_to_model_label_map: Optional[Dict[int, int]] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates causal ablation and steering metrics across a subset of the dataset.

    Computes:
      - Mean Relative Logit Drop (%)
      - Mean Steered Logit Increase (%)
            - Accuracy Drop (%) (only when model/dataset label spaces are aligned)
    """
    model_wrapper.model.eval()

    total_images = 0
    accuracy_eval_images = 0
    baseline_correct = 0
    ablated_correct = 0
    low_confidence_count = 0

    relative_drops: List[float] = []
    steer_increases: List[float] = []

    # Check if model supports standard classification (has a classification head or trained probe)
    has_classifier = (
        hasattr(model_wrapper.model, "classifier")
        or hasattr(model_wrapper.model, "logits")
        or "Classification" in model_wrapper.model.__class__.__name__
        or getattr(model_wrapper, "probe", None) is not None
    )
    can_compute_accuracy = has_classifier and labels_aligned

    if has_classifier and not labels_aligned:
        logger.warning(
            "Skipping accuracy-drop computation because dataset labels are not aligned "
            "with the model output label space."
        )

    for batch in dataloader:
        if total_images >= max_images:
            break

        images, labels = batch
        batch_size = images.shape[0]

        # Limit batch size if it exceeds the remaining allowed image count
        if total_images + batch_size > max_images:
            limit = max_images - total_images
            images = images[:limit]
            labels = labels[:limit]
            batch_size = limit

        images = images.to(device)
        labels = labels.to(device)
        total_images += batch_size

        # 1. Baseline predictions
        with torch.no_grad():
            outputs_baseline = model_wrapper.model(images)
            logits_baseline = model_wrapper.get_logits(outputs_baseline)

        # 2. Ablated predictions (ablation_strength = 0.0)
        logits_ablated = perform_causal_intervention(
            model_wrapper,
            sae,
            images,
            layer_idx,
            feature_idx,
            intervention_type="ablation",
            ablation_strength=0.0,
            target_type=target_type,
        )

        # 3. Steered predictions (steering_factor = steering_factor)
        logits_steered = perform_causal_intervention(
            model_wrapper,
            sae,
            images,
            layer_idx,
            feature_idx,
            intervention_type="steering",
            steering_factor=steering_factor,
            target_type=target_type,
        )

        # For each image in the batch, compute relative logit drop and steering increase
        # based on the class the model originally predicted
        for i in range(batch_size):
            pred_class = logits_baseline[i].argmax(dim=-1).item()

            if can_compute_accuracy:
                true_label = int(labels[i].item())
                if dataset_to_model_label_map is not None:
                    true_label = dataset_to_model_label_map.get(true_label)

                if true_label is not None:
                    accuracy_eval_images += 1
                    if pred_class == true_label:
                        baseline_correct += 1
                    pred_ablated = logits_ablated[i].argmax(dim=-1).item()
                    if pred_ablated == true_label:
                        ablated_correct += 1

            baseline_logit = logits_baseline[i, pred_class].item()
            ablated_logit = logits_ablated[i, pred_class].item()
            steered_logit = logits_steered[i, pred_class].item()

            if abs(baseline_logit) < min_baseline_abs:
                low_confidence_count += 1

            rld = (baseline_logit - ablated_logit) / (abs(baseline_logit) + 1e-8)
            sli = (steered_logit - baseline_logit) / (abs(baseline_logit) + 1e-8)

            relative_drops.append(rld * 100.0)
            steer_increases.append(sli * 100.0)

    mean_rld = np.mean(relative_drops) if relative_drops else 0.0
    mean_sli = np.mean(steer_increases) if steer_increases else 0.0
    low_confidence_fraction = (
        (low_confidence_count / total_images) if total_images > 0 else 0.0
    )

    if can_compute_accuracy and accuracy_eval_images > 0:
        base_acc = (baseline_correct / accuracy_eval_images) * 100.0
        abl_acc = (ablated_correct / accuracy_eval_images) * 100.0
        acc_drop = base_acc - abl_acc
    else:
        base_acc = None
        abl_acc = None
        acc_drop = None

    return {
        "mean_rld": mean_rld,
        "mean_sli": mean_sli,
        "relative_drops": relative_drops,
        "steer_increases": steer_increases,
        "baseline_accuracy": base_acc,
        "ablated_accuracy": abl_acc,
        "accuracy_drop": acc_drop,
        "total_images": total_images,
        "low_confidence_fraction": low_confidence_fraction,
    }
