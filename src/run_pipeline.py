import os
os.environ['MPLBACKEND'] = 'agg'
import csv
import argparse
import logging
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from typing import List, Dict, Any, Tuple

from model_loader import set_seed, ViTModelWrapper, ActivationHook
from sae import SparseAutoencoder
from caching_and_training import TokenActivationBuffer, train_sae
from interpretability import get_top_activating_patches, CLIPAutoLabeler, save_feature_grid_visualization
from causal_eval import evaluate_relative_logit_drop, plot_dose_response, perform_causal_intervention

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("run_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="Mechanistic Interpretability Pipeline with Sparse Autoencoders")
    parser.add_argument(
        "--model", 
        type=str, 
        default="google/vit-base-patch16-224",
        choices=["google/vit-base-patch16-224", "facebook/dinov2-base"],
        help="Hugging Face model checkpoint backbone"
    )
    parser.add_argument(
        "--layer", 
        type=int, 
        default=5, 
        help="0-indexed layers targeting block depth (5 = Layer 6, 10 = Layer 11)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1, 
        help="Number of training epochs for SAE"
    )
    parser.add_argument(
        "--l1_coeff", 
        type=float, 
        default=1e-3, 
        help="L1 coefficient scaling weight of feature sparsity"
    )
    parser.add_argument(
        "--expansion", 
        type=int, 
        default=8, 
        help="Overcomplete expansion ratio (hidden_dim = F * d_model)"
    )
    parser.add_argument(
        "--subset_size", 
        type=int, 
        default=1000, 
        help="Subset size for testing"
    )
    parser.add_argument(
        "--feature_idx", 
        type=int, 
        default=100, 
        help="SAE hidden feature index to analyze & causally ablate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on ('cpu', 'cuda')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "imagewoof", "imagenet"],
        help="Target dataset paradigm ('cifar10', 'imagewoof', 'imagenet')"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data",
        help="Local file path directory for custom datasets"
    )
    return parser.parse_args()

def get_top_active_features(
    model_wrapper: ViTModelWrapper,
    sae: SparseAutoencoder,
    dataloader: DataLoader,
    layer_idx: int,
    num_features: int = 10,
    target_type: str = "mlp",
    device: str = "cpu"
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
            # discard CLS token (index 0)
            patch_tokens = activation[:, 1:, :]
            f = sae.encode(patch_tokens) # [batch, num_patches, hidden_dim]
            total_activations += f.sum(dim=(0, 1))
            total_tokens += patch_tokens.shape[0] * patch_tokens.shape[1]
            
    hook.remove()
    mean_activations = total_activations / (total_tokens + 1e-8)
    top_values, top_indices = torch.topk(mean_activations, k=num_features)
    return top_indices.cpu().tolist()

def main():
    args = parse_args()
    
    # 1. Reproducibility & Device Configuration
    set_seed(42)
    device = args.device
    logger.info(f"Running pipeline on target device: {device}")
    
    # 2. Loading ViT Backbone
    model_wrapper = ViTModelWrapper(model_name=args.model, device=device)
    
    # 3. Data Loader setup
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    logger.info("Setting up CIFAR-10 test dataset...")
    dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    subset = torch.utils.data.Subset(dataset, range(args.subset_size))
    if args.dataset == "cifar10":
        logger.info("Setting up CIFAR-10 test dataset...")
        dataset = CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform)
    else:
        target_path = args.dataset_path
        if target_path == "./data":
            target_path = os.path.join(args.dataset_path, args.dataset)
            
        logger.info(f"Setting up {args.dataset} dataset from path {target_path}...")
        from torchvision.datasets import ImageFolder
        if os.path.exists(target_path):
            try:
                dataset = ImageFolder(root=target_path, transform=transform)
            except Exception as e:
                logger.warning(
                    f"Failed to load ImageFolder at {target_path} due to: {e}. "
                    "Falling back to downloading CIFAR-10."
                )
                dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
        else:
            logger.warning(f"Dataset path {target_path} not found. Falling back to downloading CIFAR-10.")
            dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
            
            
    subset = torch.utils.data.Subset(dataset, range(min(args.subset_size, len(dataset))))
    dataloader = DataLoader(subset, batch_size=8, shuffle=False)
    
    # 4. Multi-Layer Comparative Loop
    layers_to_compare = [5, 10] # Layer 6 and Layer 11
    layer_comparison_summary = []
    discovered_features_detail = []
    grid_features_dict = {} # hold Layer 11 exemplar grids
    
    # pre-load CLIP Auto-Labeler once to avoid reloading for every feature/layer
    labeler = CLIPAutoLabeler(device=device)
    candidate_concepts = [
        "sky", "grass", "metal texture", "fur", "eye", 
        "wheel", "red color", "blue color", "spotted pattern", 
        "striped pattern", "wing", "smooth surface"
    ]
    
    # we select the first image in dataset for visual intervention evaluation
    first_img_tensor, first_img_label = dataset[0]
    eval_image = first_img_tensor.unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]
    
    
    with torch.no_grad():
        out = model_wrapper.model(eval_image) # baseline class prediction for eval_image
        logits = out.logits if hasattr(out, "logits") else out.last_hidden_state[:, 0]
        predicted_class_idx = logits.argmax(dim=-1).item()
    
    logger.info(f"Image 0 predicted class index: {predicted_class_idx} (Label index: {first_img_label})")
    
    for layer in layers_to_compare:
        layer_name = f"Layer {layer+1}"
        logger.info(f"\n" + "="*50 + f"\nPROCESSING LAYER: {layer_name}\n" + "="*50)
        
        activation_buffer = TokenActivationBuffer(
            model_wrapper=model_wrapper,
            dataloader=dataloader,
            layer_idx=layer,
            target_type="mlp",
            buffer_size=8192,
            sae_batch_size=256,
            device=device
        )
        sae = SparseAutoencoder(
            d_model=model_wrapper.d_model,
            expansion_factor=args.expansion,
            tied=False
        ).to(device)
        
        # train loop
        history = train_sae(
            sae=sae,
            activation_buffer=activation_buffer,
            epochs=args.epochs,
            l1_coeff=args.l1_coeff,
            lr=1e-3,
            device=device
        )
        
        final_r2 = history[-1]["r2"] if history else 0.0
        final_l0 = history[-1]["l0"] if history else 0.0
        
        # top 10 most active features for this layer
        top_features = get_top_active_features(
            model_wrapper=model_wrapper,
            sae=sae,
            dataloader=dataloader,
            layer_idx=layer,
            num_features=10,
            target_type="mlp",
            device=device
        )
        
        logger.info(f"Top 10 active features identified for {layer_name}: {top_features}")
        
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
                device=device
            )
            
            # auto-label with CLIP
            best_concept, best_score, all_scores = labeler.label_feature(exemplars, candidate_concepts)
            
            # extract baseline logit
            with torch.no_grad():
                out_baseline = model_wrapper.model(eval_image)
                logits_base = out_baseline.logits if hasattr(out_baseline, "logits") else out_baseline.last_hidden_state[:, 0]
                baseline_logit = logits_base[:, predicted_class_idx].mean().item()
                
            # Ablation intervention
            logits_ablated = perform_causal_intervention(
                model_wrapper, sae, eval_image, layer_idx=layer, feature_idx=f_idx,
                intervention_type="ablation", scaling_factor=0.0, target_type="mlp"
            )
            ablated_logit = logits_ablated[:, predicted_class_idx].mean().item()
            rel_drop = (baseline_logit - ablated_logit) / (abs(baseline_logit) + 1e-8) * 100
            layer_drops.append(rel_drop)
            
            # Steering intervention
            logits_steered = perform_causal_intervention(
                model_wrapper, sae, eval_image, layer_idx=layer, feature_idx=f_idx,
                intervention_type="steering", scaling_factor=5.0, target_type="mlp"
            )
            steered_logit = logits_steered[:, predicted_class_idx].mean().item()
            steer_increase = (steered_logit - baseline_logit) / (abs(baseline_logit) + 1e-8) * 100
            
            feat_result = {
                "Feature Index": f_idx,
                "Target Layer": layer_name,
                "Assigned CLIP Concept": best_concept,
                "CLIP Confidence Score": f"{best_score:.4f}",
                "Baseline Logit": f"{baseline_logit:.4f}",
                "Ablated Logit": f"{ablated_logit:.4f}",
                "Relative Logit Drop (%)": f"{rel_drop:.4f}",
                "Steered Logit Increase (%)": f"{steer_increase:.4f}"
            }
            discovered_features_detail.append(feat_result)
            
            # here we cache Layer 11 representative features (top 5) for grid visualization
            if layer == 10 and len(grid_features_dict) < 5:
                grid_features_dict[f_idx] = {
                    "exemplars": exemplars,
                    "concept": best_concept
                }
                
        # this is the mean relative logit drop for the layer
        mean_logit_drop = sum(layer_drops) / (len(layer_drops) + 1e-8)
        
        layer_comparison_summary.append({
            "Layer": layer_name,
            "R^2 Score": f"{final_r2:.4f}",
            "L_0 Norm": f"{final_l0:.2f}",
            "Mean Logit Drop": f"{mean_logit_drop:.4f}%"
        })
        
    # 5. Export comparative table to stdout and file
    linea_div = "=" * 59
    md_table_title = "             MULTI-LAYER COMPARATIVE SUMMARY"
    md_table_header = f"{linea_div}\n{md_table_title}\n{linea_div}"
    print(md_table_header)
    markdown_table = (
        f"| {'Layer':10} | {'R^2 Score':9} | {'L_0 Norm':8} | {'Mean Logit Drop':17} |\n"
        "|" + "-"*12 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*19 + "|\n"
    )
    for row in layer_comparison_summary:
        markdown_table += f"| {row['Layer']:10} | {row['R^2 Score']:9} | {row['L_0 Norm']:8} | {row['Mean Logit Drop']:17} |\n"
    markdown_table += "="*59 + "\n"
    
    print(markdown_table)
    
    with open("layer_comparison_summary.md", mode="w", encoding="utf-8") as f:
        f.write(markdown_table)
    logger.info("Saved layer comparison table to layer_comparison_summary.md")
    
    # 6. Save unified 5x5 feature grid visualization for Layer 11
    if grid_features_dict:
        save_feature_grid_visualization(grid_features_dict, "multi_feature_exemplar_grid.png")
        
    # 7. Quantitative CSV Exporter
    csv_file_path = "discovered_features_summary.csv"
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Feature Index", "Target Layer", "Assigned CLIP Concept", 
            "CLIP Confidence Score", "Baseline Logit", "Ablated Logit", 
            "Relative Logit Drop (%)", "Steered Logit Increase (%)"
        ])
        writer.writeheader()
        writer.writerows(discovered_features_detail)
    logger.info(f"Saved quantitative summary to {csv_file_path}")
    
    # 8. Plot Dose-Response for a representative feature (Feature 100 on Layer 11)
    logger.info("Generating Causal Dose-Response curve for representation...")
    plot_dose_response(
        model_wrapper=model_wrapper,
        sae=sae,
        images=eval_image,
        layer_idx=10,
        feature_idx=top_features[0] if top_features else 100,
        target_class_idx=predicted_class_idx,
        target_type="mlp",
        save_path="dose_response_curve.png"
    )
    
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
