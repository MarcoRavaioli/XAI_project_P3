import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Callable, Optional
from transformers import ViTModel, Dinov2Model, ViTImageProcessor, AutoImageProcessor

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
    def __init__(self, module: nn.Module, callback: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.module = module
        self.callback = callback
        self.handle = None
        self.activation = None

    def hook_fn(self, module: nn.Module, inputs: tuple, outputs: Any) -> Any:
        # If layer returns tuple (attention + metadata), unpack
        is_tuple = isinstance(outputs, tuple)
        actual_output = outputs[0] if is_tuple else outputs
        
        self.activation = actual_output.detach() # no memory leaks pls
        
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
    caching activation streams across layers 6 and 11.
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: str = "cpu"):
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
            self.model = ViTModel.from_pretrained(model_name)
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model_type = "vit"
            self.patch_size = 16
            self.grid_size = 14
            
        self.model.to(self.device)
        self.model.eval()
        
        self.d_model = self.model.config.hidden_size
        logger.info(f"Model loaded. Type: {self.model_type}, Patch Size: {self.patch_size}, Grid Size: {self.grid_size}, d_model: {self.d_model}")

    def get_layer(self, layer_idx: int) -> nn.Module:
        """
        Retrieves the block module at layer_idx.
        layer_idx: 0-indexed (5 for Layer 6, 10 for Layer 11).
        """
        # some models have layers directly on the root module
        if hasattr(self.model, "layers"):
            return self.model.layers[layer_idx]
        layer_container = getattr(self.model, self.model_type, self.model)
        try:
            return layer_container.encoder.layer[layer_idx]
        except AttributeError: # fallback to direct model.encoder.layer search
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
                raise AttributeError(f"Could not locate MLP submodule in layer {layer_idx}")
        elif target_type == "residual":
            return layer
        else:
            raise ValueError(f"Unknown target_type: {target_type}. Must be 'mlp' or 'residual'")
