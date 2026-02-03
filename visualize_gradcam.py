"""
GradCAM Visualization Script for PromptSG Model

This script generates GradCAM visualizations for the PromptSG person re-identification model.
It supports multiple visualization methods:
1. GradCAM - Standard gradient-weighted class activation mapping
2. GradCAM++ - Improved GradCAM with better localization
3. MIM Attention - Direct visualization of Multimodal Interaction Module attention

Usage:
    python visualize_gradcam.py --config_file configs/person/vit_promptsg.yml \
                                --weight path/to/model.pth \
                                --image_path path/to/image.jpg \
                                --method gradcam \
                                --output_dir ./gradcam_outputs

    # For batch visualization from dataset:
    python visualize_gradcam.py --config_file configs/person/vit_promptsg.yml \
                                --weight path/to/model.pth \
                                --batch_mode \
                                --num_samples 20 \
                                --method mim_attention
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import cfg
from model.make_model_promptsg import make_model
from datasets.make_dataloader_promptsg import make_dataloader
from utils.gradcam import (
    GradCAM,
    GradCAMPlusPlus,
    MIMAttentionCAM,
    visualize_gradcam,
    batch_visualize_gradcam,
    denormalize_image,
    overlay_heatmap
)
from datasets.preprocessing import get_transforms


def get_target_layers(model, layer_type='mim'):
    """
    Get target layers for GradCAM based on layer type.

    Args:
        model: PromptSG model
        layer_type: Type of layer to target
            - 'mim': Multimodal Interaction Module cross-attention
            - 'vit_last': Last ViT transformer block
            - 'vit_proj': ViT projection layer
            - 'classifier': Classification head
            - 'bottleneck': Bottleneck layer

    Returns:
        List of target layers
    """
    target_layers = []

    if layer_type == 'mim':
        # Target MIM cross-attention
        if hasattr(model, 'mim'):
            if hasattr(model.mim, 'cross_attn'):
                target_layers.append(model.mim.cross_attn)
            # Also add post blocks
            if hasattr(model.mim, 'post_blocks'):
                target_layers.append(model.mim.post_blocks[-1])
        else:
            print("Warning: MIM module not found")

    elif layer_type == 'vit_last':
        # Target last ViT transformer block
        if hasattr(model, 'image_encoder'):
            if hasattr(model.image_encoder, 'transformer'):
                # For ViT
                if hasattr(model.image_encoder.transformer, 'resblocks'):
                    target_layers.append(model.image_encoder.transformer.resblocks[-1])
            elif hasattr(model.image_encoder, 'layer4'):
                # For ResNet
                target_layers.append(model.image_encoder.layer4)

    elif layer_type == 'vit_proj':
        # Target projection layer
        if hasattr(model, 'image_encoder'):
            if hasattr(model.image_encoder, 'ln_post'):
                target_layers.append(model.image_encoder.ln_post)

    elif layer_type == 'classifier':
        # Target classifier
        if hasattr(model, 'classifier_proj'):
            target_layers.append(model.classifier_proj)
        elif hasattr(model, 'classifier'):
            target_layers.append(model.classifier)

    elif layer_type == 'bottleneck':
        # Target bottleneck
        if hasattr(model, 'bottleneck_proj'):
            target_layers.append(model.bottleneck_proj)

    elif layer_type == 'inversion':
        # Target inversion network
        if hasattr(model, 'inversion'):
            target_layers.append(model.inversion.fc3)

    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

    if not target_layers:
        print(f"Warning: No layers found for type '{layer_type}'")
        # Fallback to bottleneck
        if hasattr(model, 'bottleneck_proj'):
            target_layers.append(model.bottleneck_proj)

    return target_layers


def load_single_image(image_path, cfg):
    """
    Load and preprocess a single image.

    Args:
        image_path: Path to image
        cfg: Config object

    Returns:
        Preprocessed image tensor
    """
    # Get transforms
    transform = get_transforms(cfg, is_train=False)

    # Load image
    image = Image.open(image_path).convert('RGB')

    # Apply transforms
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def visualize_all_methods(
    model,
    image_tensor,
    output_dir,
    image_name='image',
    device='cuda'
):
    """
    Generate visualizations using all available methods.

    Args:
        model: PromptSG model
        image_tensor: Input image tensor
        output_dir: Output directory
        image_name: Base name for output files
        device: Device to use
    """
    os.makedirs(output_dir, exist_ok=True)

    model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)

    results = {}

    # 1. MIM Attention (direct attention visualization)
    print("Generating MIM Attention visualization...")
    try:
        mim_cam = MIMAttentionCAM(model)
        heatmap_mim = mim_cam(image_tensor)
        mim_cam.remove_hooks()

        original_image = denormalize_image(image_tensor)
        overlay_mim = overlay_heatmap(original_image, heatmap_mim)

        results['mim_attention'] = {
            'heatmap': heatmap_mim,
            'overlay': overlay_mim
        }
    except Exception as e:
        print(f"  Error: {e}")

    # 2. GradCAM on MIM
    print("Generating GradCAM on MIM...")
    try:
        target_layers = get_target_layers(model, 'mim')
        if target_layers:
            gradcam = GradCAM(model, target_layers)
            heatmap_gradcam = gradcam(image_tensor)
            gradcam.remove_hooks()

            original_image = denormalize_image(image_tensor)
            overlay_gradcam = overlay_heatmap(original_image, heatmap_gradcam)

            results['gradcam_mim'] = {
                'heatmap': heatmap_gradcam,
                'overlay': overlay_gradcam
            }
    except Exception as e:
        print(f"  Error: {e}")

    # 3. GradCAM on ViT last layer
    print("Generating GradCAM on ViT last layer...")
    try:
        target_layers = get_target_layers(model, 'vit_last')
        if target_layers:
            gradcam = GradCAM(model, target_layers)
            heatmap_vit = gradcam(image_tensor)
            gradcam.remove_hooks()

            original_image = denormalize_image(image_tensor)
            overlay_vit = overlay_heatmap(original_image, heatmap_vit)

            results['gradcam_vit'] = {
                'heatmap': heatmap_vit,
                'overlay': overlay_vit
            }
    except Exception as e:
        print(f"  Error: {e}")

    # 4. GradCAM++ on MIM
    print("Generating GradCAM++ on MIM...")
    try:
        target_layers = get_target_layers(model, 'mim')
        if target_layers:
            gradcam_pp = GradCAMPlusPlus(model, target_layers)
            heatmap_pp = gradcam_pp(image_tensor)
            gradcam_pp.remove_hooks()

            original_image = denormalize_image(image_tensor)
            overlay_pp = overlay_heatmap(original_image, heatmap_pp)

            results['gradcam_pp'] = {
                'heatmap': heatmap_pp,
                'overlay': overlay_pp
            }
    except Exception as e:
        print(f"  Error: {e}")

    # Create combined visualization
    original_image = denormalize_image(image_tensor)

    num_methods = len(results) + 1  # +1 for original
    fig, axes = plt.subplots(2, num_methods, figsize=(4 * num_methods, 8))

    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')

    # Plot each method
    for idx, (method_name, method_results) in enumerate(results.items(), 1):
        axes[0, idx].imshow(method_results['heatmap'], cmap='jet')
        axes[0, idx].set_title(f'{method_name}\nHeatmap')
        axes[0, idx].axis('off')

        axes[1, idx].imshow(method_results['overlay'])
        axes[1, idx].set_title(f'{method_name}\nOverlay')
        axes[1, idx].axis('off')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f'{image_name}_all_methods.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined visualization to {save_path}")

    plt.close()

    # Save individual results
    for method_name, method_results in results.items():
        # Save heatmap
        heatmap_path = os.path.join(output_dir, f'{image_name}_{method_name}_heatmap.png')
        plt.figure(figsize=(8, 8))
        plt.imshow(method_results['heatmap'], cmap='jet')
        plt.axis('off')
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save overlay
        overlay_path = os.path.join(output_dir, f'{image_name}_{method_name}_overlay.png')
        Image.fromarray(method_results['overlay']).save(overlay_path)

    return results


def main():
    parser = argparse.ArgumentParser(description='GradCAM Visualization for PromptSG')

    # Config
    parser.add_argument('--config_file', default='configs/person/vit_promptsg.yml',
                        type=str, help='Path to config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Additional config options')

    # Model
    parser.add_argument('--weight', type=str, required=True,
                        help='Path to model weights')

    # Input
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to single image (for single image mode)')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Use batch mode with dataloader')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples in batch mode')

    # Visualization
    parser.add_argument('--method', type=str, default='all',
                        choices=['gradcam', 'gradcam++', 'mim_attention', 'all'],
                        help='Visualization method')
    parser.add_argument('--layer_type', type=str, default='mim',
                        choices=['mim', 'vit_last', 'vit_proj', 'classifier', 'bottleneck', 'inversion'],
                        help='Target layer type for GradCAM')

    # Output
    parser.add_argument('--output_dir', type=str, default='./gradcam_outputs',
                        help='Output directory')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    # Load config
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataloader for getting num_classes, camera_num, view_num
    print("Loading dataloader...")
    train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # Create model
    print("Creating model...")
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # Load weights
    print(f"Loading weights from {args.weight}...")
    state_dict = torch.load(args.weight, map_location='cpu')

    # Handle DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully. Using device: {device}")

    if args.batch_mode:
        # Batch visualization mode
        print(f"\nRunning batch visualization for {args.num_samples} samples...")

        target_layers = get_target_layers(model, args.layer_type)

        if args.method == 'all':
            # For batch mode with 'all', use MIM attention as default
            method = 'mim_attention'
        else:
            method = args.method

        batch_visualize_gradcam(
            model=model,
            dataloader=val_loader,
            target_layers=target_layers,
            num_samples=args.num_samples,
            method=method,
            output_dir=args.output_dir,
            device=device
        )

    elif args.image_path:
        # Single image mode
        print(f"\nProcessing single image: {args.image_path}")

        image_tensor = load_single_image(args.image_path, cfg)
        image_name = os.path.splitext(os.path.basename(args.image_path))[0]

        if args.method == 'all':
            # Generate all visualizations
            visualize_all_methods(
                model=model,
                image_tensor=image_tensor,
                output_dir=args.output_dir,
                image_name=image_name,
                device=device
            )
        else:
            # Single method
            target_layers = get_target_layers(model, args.layer_type)

            save_path = os.path.join(args.output_dir, f'{image_name}_{args.method}.png')

            visualize_gradcam(
                model=model,
                image_tensor=image_tensor,
                target_layers=target_layers,
                method=args.method,
                save_path=save_path,
                show=False
            )

    else:
        # Use first image from validation set
        print("\nNo image path provided. Using first image from validation set...")

        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                pids = batch[1] if len(batch) > 1 else None
            else:
                images = batch
                pids = None
            break

        image_tensor = images[0:1].to(device)

        if args.method == 'all':
            visualize_all_methods(
                model=model,
                image_tensor=image_tensor,
                output_dir=args.output_dir,
                image_name='sample_0',
                device=device
            )
        else:
            target_layers = get_target_layers(model, args.layer_type)

            save_path = os.path.join(args.output_dir, f'sample_0_{args.method}.png')

            visualize_gradcam(
                model=model,
                image_tensor=image_tensor,
                target_layers=target_layers,
                method=args.method,
                save_path=save_path,
                show=False
            )

    print(f"\nVisualization complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
