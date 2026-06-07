import os

os.environ["MPLBACKEND"] = "agg"
import logging
import torch
from typing import Tuple
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from model_loader import ViTModelWrapper, ActivationHook
from sae import SparseAutoencoder

logger = logging.getLogger(__name__)


def perform_causal_intervention(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    images: torch.Tensor,
    layer_idx: int,
    feature_idx: int,
    intervention_type: str = "ablation",
    scaling_factor: float = 0.0,
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

    submodule = model_wrapper.get_submodule(layer_idx, target_type) # target submodule

    def intervention_callback(x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        cls_token = x[
            :, 0:1, :
        ]  # keep gobal classification token untouched
        patch_tokens = x[
            :, 1:, :
        ]  # [batch, num_patches, d_model]

        # now we project to SAE space to obtain hidden activations
        f = sae.encode(patch_tokens)  # [batch, num_patches, hidden_dim]
        f_j = f[:, :, feature_idx]  # Target feature activation: [batch, num_patches]

        # we then retrieve target feature's reconstruction column vector from decoder
        W_dec_j = sae.W_dec[:, feature_idx]  # [d_model]

        if intervention_type == "ablation":
            # Subtract only the visual feature projection from the residual stream
            # scaling_factor 0.0 is full zeroing, 0.5 leaves half the activation
            ablation_strength = 1.0 - scaling_factor
            patch_tokens_modified = patch_tokens - ablation_strength * f_j.unsqueeze(
                -1
            ) * W_dec_j.view(1, 1, -1)

        elif intervention_type == "steering":
            # Scale up target feature's activation
            # patch_steered = patch + (scaling_factor - 1.0) * f_j * W_dec_j
            patch_tokens_modified = patch_tokens + (
                scaling_factor - 1.0
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
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                # if Dinov2Model is loaded, outputs.last_hidden_state[:, 0] is the CLS token
                logits = outputs.last_hidden_state[:, 0]
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
        if hasattr(outputs_baseline, "logits"):
            logits_baseline = outputs_baseline.logits
        else:
            logits_baseline = outputs_baseline.last_hidden_state[:, 0]

        baseline_logit = logits_baseline[:, target_class_idx].mean().item()

    # Ablated logit (scaling_factor = 0.0)
    logits_ablated = perform_causal_intervention(
        model_wrapper,
        sae,
        images,
        layer_idx,
        feature_idx,
        intervention_type="ablation",
        scaling_factor=0.0,
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
        if hasattr(outputs_baseline, "logits"):
            logits_baseline = outputs_baseline.logits
        else:
            logits_baseline = outputs_baseline.last_hidden_state[:, 0]
        baseline_logit = logits_baseline[:, target_class_idx].mean().item()

    for strength in ablation_strengths:
        logits_intervened = perform_causal_intervention(
            model_wrapper,
            sae,
            images,
            layer_idx,
            feature_idx,
            intervention_type="ablation",
            scaling_factor=strength,
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
    percent_ablation = (1.0 - ablation_strengths) * 100 # so 100% ablation is at scaling_factor=0.0

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
                epochs, values,
                marker="o", linewidth=2, markersize=5,
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