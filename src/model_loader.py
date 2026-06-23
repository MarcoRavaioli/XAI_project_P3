import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Callable, Optional
from torch.utils.data import DataLoader
from transformers import (
    ViTForImageClassification,
    Dinov2Model,
    ViTImageProcessor,
    AutoImageProcessor,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Global seed for reproducibility across torch, numpy, and random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # we need to ensure deterministic behavior as much as possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")


class ActivationHook:
    """
    Reusable PyTorch forward hook class to cache intermediate activations
    or to modify activations dynamically during a forward pass.

    Citations:
        - Elhage et al. [2021] ("A Mathematical Framework for Transformer Circuits")
    """

    def __init__(
        self,
        module: nn.Module,
        callback: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.module = module
        self.callback = callback
        self.handle = None
        self.activation = None

    def hook_fn(self, module: nn.Module, inputs: tuple, outputs: Any) -> Any:
        # If layer returns tuple (attention + metadata), unpack
        is_tuple = isinstance(outputs, tuple)
        actual_output = outputs[0] if is_tuple else outputs

        self.activation = actual_output.detach()  # no memory leaks pls

        # apply eventual intervention callbacks
        if self.callback is not None:
            modified_output = self.callback(actual_output)
            if is_tuple:
                return (modified_output,) + outputs[1:]
            return modified_output

        return outputs

    def register(self):
        """Registers the hook to the targeted submodule."""
        if self.handle is None:
            self.handle = self.module.register_forward_hook(self.hook_fn)
            logger.debug(f"Hook registered on {self.module.__class__.__name__}")

    def remove(self):
        """Removes the hook from the submodule."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            logger.debug("Hook removed successfully")


class ViTModelWrapper:
    """
    Wrapper around pre-trained Hugging Face Vision Transformers (ViT & DINOv2)
    to simplify backbone loading, dynamic patch size configuration, and
    caching activation streams across arbitrary layers.
    """

    def __init__(
        self, model_name: str = "google/vit-base-patch16-224", device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device

        logger.info(f"Initializing {model_name} on {device}...")

        if "dinov2" in model_name:
            self.model = Dinov2Model.from_pretrained(model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model_type = "dinov2"
            self.patch_size = 14
            self.grid_size = 16
        else:
            self.model = ViTForImageClassification.from_pretrained(model_name)
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model_type = "vit"
            self.patch_size = 16
            self.grid_size = 14

        self.model.to(self.device)
        self.model.eval()

        self.d_model = self.model.config.hidden_size
        self.probe = None
        logger.info(
            f"Model loaded. Type: {self.model_type}, Patch Size: {self.patch_size}, Grid Size: {self.grid_size}, d_model: {self.d_model}"
        )

    def get_layer(self, layer_idx: int) -> nn.Module:
        """
        Retrieves the block module at layer_idx.
        layer_idx: 0-indexed (5 for Layer 6, 10 for Layer 11).
        """
        # ViTForImageClassification nests encoder under .vit, Dinov2Model under .dinov2
        for backbone_attr in ["vit", "dinov2"]:
            backbone = getattr(self.model, backbone_attr, None)
            if backbone is None:
                continue
            # Newer transformers: backbone.layers (ModuleList directly)
            if hasattr(backbone, "layers"):
                return backbone.layers[layer_idx]
            # Older transformers: backbone.encoder.layer
            if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
                return backbone.encoder.layer[layer_idx]

        # Fallback for raw encoder models
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            return self.model.encoder.layer[layer_idx]

        raise AttributeError(f"Could not resolve layer pathways in {self.model_name}")

    def get_submodule(self, layer_idx: int, target_type: str = "mlp") -> nn.Module:
        """
        Retrieves either the MLP output or the whole block (residual stream) at layer_idx.
        target_type: 'mlp' or 'residual'
        """
        layer = self.get_layer(layer_idx)
        if target_type == "mlp":
            # unified models (use .mlp directly) and legacy support
            if hasattr(layer, "mlp"):
                return layer.mlp
            elif hasattr(layer, "output") and hasattr(layer.output, "dense"):
                return layer.output.dense
            else:
                raise AttributeError(
                    f"Could not locate MLP submodule in layer {layer_idx}"
                )
        elif target_type == "residual":
            return layer
        else:
            raise ValueError(
                f"Unknown target_type: {target_type}. Must be 'mlp' or 'residual'"
            )

    def train_linear_probe(self, dataloader: DataLoader):
        """
        Trains and registers a linear probe for models without a classification head.
        """
        probe = train_linear_probe(self, dataloader, self.device)
        probe.to(self.device)
        probe.eval()
        self.probe = probe

    def get_logits(self, outputs: Any) -> torch.Tensor:
        """
        Extracts logits from model outputs, using the trained linear probe
        as a fallback for models without a classification head.
        """
        if hasattr(outputs, "logits"):
            return outputs.logits
        elif self.probe is not None:
            if hasattr(outputs, "last_hidden_state"):
                cls_emb = outputs.last_hidden_state[:, 0]
            else:
                cls_emb = outputs[0][:, 0]
            return self.probe(cls_emb)
        else:
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state[:, 0]
            else:
                return outputs[0][:, 0]


def get_num_classes(dataset) -> int:
    """
    Statically extracts the number of classes from CIFAR10, ImageFolder, or Hugging Face Dataset.
    """
    base_dataset = dataset
    while hasattr(base_dataset, "dataset"):
        if isinstance(base_dataset, torch.utils.data.Subset):
            base_dataset = base_dataset.dataset
        else:
            break

    if hasattr(base_dataset, "classes") and base_dataset.classes is not None:
        return len(base_dataset.classes)

    if hasattr(base_dataset, "dataset") and hasattr(base_dataset.dataset, "features"):
        features = base_dataset.dataset.features
        if "label" in features and hasattr(features["label"], "num_classes"):
            return features["label"].num_classes

    if hasattr(base_dataset, "dataset") and hasattr(base_dataset.dataset, "info") and hasattr(base_dataset.dataset.info, "features"):
        features = base_dataset.dataset.info.features
        if "label" in features and hasattr(features["label"], "num_classes"):
            return features["label"].num_classes

    try:
        labels = [base_dataset[i][1] for i in range(min(100, len(base_dataset)))]
        return max(labels) + 1
    except Exception:
        return 10


def train_linear_probe(
    model_wrapper: "ViTModelWrapper",
    dataloader: DataLoader,
    device: str,
) -> nn.Linear:
    """
    Trains a linear probe on top of frozen CLS embeddings from the vision transformer backbone.
    """
    model_wrapper.model.eval()
    
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            batch_labels = batch[1].to(device)
            
            outputs = model_wrapper.model(images)
            if hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state[:, 0]
            elif hasattr(outputs, "logits"):
                emb = outputs.logits
            else:
                emb = outputs[0][:, 0]
            
            embeddings.append(emb.cpu())
            labels.append(batch_labels.cpu())
            
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    num_classes = get_num_classes(dataloader.dataset)
    d_model = embeddings.shape[1]
    
    logger.info(f"Training linear probe on {embeddings.shape[0]} samples with {num_classes} classes...")
    
    probe = nn.Linear(d_model, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    probe.train()
    dataset_size = embeddings.shape[0]
    batch_size = min(32, dataset_size)
    
    for epoch in range(10):
        indices = torch.randperm(dataset_size)
        epoch_loss = 0.0
        correct = 0
        
        for start_idx in range(0, dataset_size, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            x_batch = embeddings[batch_idx].to(device)
            y_batch = labels[batch_idx].to(device)
            
            optimizer.zero_grad()
            preds = probe(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.shape[0]
            correct += (preds.argmax(dim=-1) == y_batch).sum().item()
            
        acc = (correct / dataset_size) * 100.0
        logger.debug(f"Linear Probe Epoch {epoch+1}/10 | Loss: {epoch_loss/dataset_size:.4f} | Accuracy: {acc:.2f}%")
        
    probe.eval()
    logger.info(f"Linear probe training complete. Final Accuracy: {acc:.2f}%")
    return probe
