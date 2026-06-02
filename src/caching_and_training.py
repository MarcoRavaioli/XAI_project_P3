import gc
import logging
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, List

from model_loader import ActivationHook, ViTModelWrapper
from sae import SparseAutoencoder

logger = logging.getLogger(__name__)


class TokenActivationBuffer:
    """
    Memory-safe streaming buffer that feeds spatial patch activations to the SAE.
    It passes image batches through a Vision Transformer, extracts activations via a hook,
    discards the outlier [CLS] token (index 0), flattens the patch activations across the sequence and batch,
    buffers them to ensure diverse representation shuffling, and yields token batches on-the-fly.

    We implemented this to avoid unrecoverable OOM errors in automated runs.

    Citations:
        - Elhage et al. [2021] ("A Mathematical Framework for Transformer Circuits")
        - Bricken et al. [2023] ("Towards Monosemanticity")
    """

    def __init__(
        self,
        model_wrapper: ViTModelWrapper,
        dataloader: DataLoader,
        layer_idx: int,
        target_type: str = "mlp",
        buffer_size: int = 65536,  # capacity in tokens (tokens = batch * num_patches)
        sae_batch_size: int = 1024,
        device: str = "cpu",
    ):
        self.model_wrapper = model_wrapper
        self.dataloader = dataloader
        self.layer_idx = layer_idx
        self.target_type = target_type
        self.buffer_size = buffer_size
        self.sae_batch_size = sae_batch_size
        self.device = device

        # the exact layer submodule to hook
        submodule = model_wrapper.get_submodule(layer_idx, target_type)
        self.hook = ActivationHook(submodule)

        # init the buffer
        self.buffer = torch.empty((0, model_wrapper.d_model), device=device)
        self.data_iter = iter(dataloader)
        self.epoch_done = False

    def fill_buffer(self):
        """
        Streams image batches through the model, intercepts activations via hooks,
        discards CLS tokens, and populates the shuffling buffer.
        """
        self.hook.register()

        while len(self.buffer) < self.buffer_size:
            try:
                batch = next(self.data_iter)
                images = batch[0].to(self.device)
            except StopIteration:
                self.epoch_done = True
                break

            with torch.no_grad():
                _ = self.model_wrapper.model(images)

            activation = self.hook.activation
            if activation is None:
                raise RuntimeError(
                    "Activation hook failed to capture tensor. Verify model forward flow."
                )

            # ViT and DINOv2 prepend a classification token [CLS] at seq index 0.
            # Discard index 0 to focus on spatial patch tokens (indices 1:).
            patch_activations = activation[
                :, 1:, :
            ]  # [batch_size, num_patches, d_model]
            flat_activations = patch_activations.reshape(-1, self.model_wrapper.d_model)
            self.buffer = torch.cat(
                [self.buffer, flat_activations.to(self.device)], dim=0
            )

        self.hook.remove()

        # we shuffle the token buffer across all batch components to avoid intra-batch correlations
        if len(self.buffer) > 0:
            perm = torch.randperm(len(self.buffer), device=self.device)
            self.buffer = self.buffer[perm]

    def next_batch(self) -> Optional[torch.Tensor]:
        """
        Returns a single batch of token activations of size sae_batch_size.
        """
        if len(self.buffer) < self.sae_batch_size and not self.epoch_done:
            self.fill_buffer()

        if len(self.buffer) == 0:
            return None

        take_size = min(len(self.buffer), self.sae_batch_size)
        batch_tokens = self.buffer[:take_size]
        self.buffer = self.buffer[take_size:]
        return batch_tokens

    def reset(self):
        """
        Resets data loaders and token buffer at epoch borders.
        """
        self.data_iter = iter(self.dataloader)
        self.epoch_done = False
        self.buffer = torch.empty((0, self.model_wrapper.d_model), device=self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache() # gc to save ram
        gc.collect()


def train_sae(
    sae: SparseAutoencoder,
    activation_buffer: TokenActivationBuffer,
    epochs: int,
    l1_coeff: float,
    lr: float = 3e-4,
    device: str = "cpu",
) -> List[Dict[str, float]]:
    """
    SAE training routine managing step updates, weight projection normalizations, and stats logging.

    Citations:
        - Cunningham et al. [2024] ("Sparse Autoencoders Find Highly Interpretable Features")
    """
    sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    history = []

    if device == "cpu" and torch.cuda.is_available():
        logger.warning(
            "CUDA is available, but the pipeline has fallback to CPU. </3 Good luck."
        )

    logger.info("Starting Sparse Autoencoder training pipeline...")

    for epoch in range(epochs):
        activation_buffer.reset()
        step = 0

        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_l1 = 0.0
        epoch_r2 = 0.0
        epoch_l0 = 0.0

        while True:
            batch_tokens = activation_buffer.next_batch()
            if batch_tokens is None:
                break

            optimizer.zero_grad() # clean grad mem

            # Forward pass: reconstruct activations
            x_hat, f = sae(batch_tokens)

            loss_dict = sae.compute_loss(batch_tokens, x_hat, f, l1_coeff)
            loss = loss_dict["loss"]

            loss.backward() # backprop
            optimizer.step()

            # project decoder weights to unit norms to avoid rescaling shortcuts
            sae.normalize_decoder_weights()

            # metrics
            x_mean = torch.mean(batch_tokens, dim=0, keepdim=True)
            total_ss = torch.sum((batch_tokens - x_mean) ** 2)
            residual_ss = torch.sum((batch_tokens - x_hat) ** 2)
            r2 = (1.0 - (residual_ss / (total_ss + 1e-8))).item()
            l0 = torch.mean(torch.sum(f > 0, dim=-1).float()).item() # sparsity

            epoch_loss += loss.item()
            epoch_mse += loss_dict["mse_loss"].item()
            epoch_l1 += loss_dict["l1_loss"].item()
            epoch_r2 += r2
            epoch_l0 += l0

            step += 1

        if step > 0:
            metrics = {
                "epoch": epoch + 1,
                "loss": epoch_loss / step,
                "mse": epoch_mse / step,
                "l1": epoch_l1 / step,
                "r2": epoch_r2 / step,
                "l0": epoch_l0 / step,
            }
            history.append(metrics)

            logger.info(
                f"Epoch {metrics['epoch']:02d}/{epochs:02d} | "
                f"Loss: {metrics['loss']:.6f} | "
                f"MSE: {metrics['mse']:.6f} | "
                f"L1: {metrics['l1']:.4f} | "
                f"R^2: {metrics['r2']:.4f} | "
                f"L0: {metrics['l0']:.2f}"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    logger.info("Sparse Autoencoder training complete.")
    return history
