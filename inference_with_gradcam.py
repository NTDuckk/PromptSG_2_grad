"""
Inference with GradCAM Visualization for PromptSG

A simplified script for running inference on images and generating
attention/GradCAM visualizations.

Usage:
    # Single image
    python inference_with_gradcam.py \
        --config configs/person/vit_promptsg.yml \
        --weights output/ViT-B-16_60.pth \
        --image query_image.jpg \
        --output_dir results/

    # Compare query with gallery
    python inference_with_gradcam.py \
        --config configs/person/vit_promptsg.yml \
        --weights output/ViT-B-16_60.pth \
        --query query_image.jpg \
        --gallery gallery_image.jpg \
        --output_dir results/
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from config import cfg
from model.make_model_promptsg import make_model
from datasets.preprocessing import get_transforms


def load_image(image_path, cfg):
    """Load and preprocess an image."""
    transform = get_transforms(cfg, is_train=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor to numpy image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.cpu().clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = tensor.clamp(0, 1)
    image = tensor.permute(1, 2, 0).numpy()
    return np.uint8(255 * image)


def visualize_attention(
    model,
    image_tensor,
    original_image,
    output_path,
    device='cuda'
):
    """
    Visualize the MIM attention map.

    Args:
        model: PromptSG model
        image_tensor: Preprocessed image tensor
        original_image: Original PIL image
        output_path: Path to save visualization
        device: Device to use
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Get attention map using model's built-in method
        attn_map = model.get_attention_map(image_tensor, reshape_to_image=True)

    # Convert to numpy
    attn_map = attn_map.cpu().numpy()[0]  # (H, W)

    # Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Resize to original image size
    original_size = (original_image.size[0], original_image.size[1])  # (W, H)
    attn_map_resized = cv2.resize(attn_map, original_size)

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Create overlay
    original_np = np.array(original_image)
    overlay = cv2.addWeighted(original_np, 0.5, heatmap, 0.5, 0)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(attn_map_resized, cmap='jet')
    axes[1].set_title('MIM Attention Map')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved attention visualization to {output_path}")

    return attn_map_resized


def compute_similarity(model, query_tensor, gallery_tensor, device='cuda'):
    """
    Compute similarity between query and gallery images.

    Args:
        model: PromptSG model
        query_tensor: Query image tensor
        gallery_tensor: Gallery image tensor
        device: Device to use

    Returns:
        Similarity score
    """
    model.eval()
    query_tensor = query_tensor.to(device)
    gallery_tensor = gallery_tensor.to(device)

    with torch.no_grad():
        query_feat = model(query_tensor)
        gallery_feat = model(gallery_tensor)

        # Normalize features
        query_feat = F.normalize(query_feat, p=2, dim=1)
        gallery_feat = F.normalize(gallery_feat, p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.mm(query_feat, gallery_feat.t()).item()

    return similarity


def compare_with_gradcam(
    model,
    query_path,
    gallery_path,
    cfg,
    output_dir,
    device='cuda'
):
    """
    Compare query and gallery images with attention visualization.

    Args:
        model: PromptSG model
        query_path: Path to query image
        gallery_path: Path to gallery image
        cfg: Config object
        output_dir: Output directory
        device: Device to use
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    query_tensor, query_img = load_image(query_path, cfg)
    gallery_tensor, gallery_img = load_image(gallery_path, cfg)

    # Compute similarity
    similarity = compute_similarity(model, query_tensor, gallery_tensor, device)

    # Get attention maps
    model.eval()
    query_tensor = query_tensor.to(device)
    gallery_tensor = gallery_tensor.to(device)

    with torch.no_grad():
        query_attn = model.get_attention_map(query_tensor, reshape_to_image=True).cpu().numpy()[0]
        gallery_attn = model.get_attention_map(gallery_tensor, reshape_to_image=True).cpu().numpy()[0]

    # Normalize attention maps
    query_attn = (query_attn - query_attn.min()) / (query_attn.max() - query_attn.min() + 1e-8)
    gallery_attn = (gallery_attn - gallery_attn.min()) / (gallery_attn.max() - gallery_attn.min() + 1e-8)

    # Resize to image sizes
    query_size = (query_img.size[0], query_img.size[1])
    gallery_size = (gallery_img.size[0], gallery_img.size[1])

    query_attn_resized = cv2.resize(query_attn, query_size)
    gallery_attn_resized = cv2.resize(gallery_attn, gallery_size)

    # Create heatmaps
    query_heatmap = cv2.applyColorMap(np.uint8(255 * query_attn_resized), cv2.COLORMAP_JET)
    query_heatmap = cv2.cvtColor(query_heatmap, cv2.COLOR_BGR2RGB)
    gallery_heatmap = cv2.applyColorMap(np.uint8(255 * gallery_attn_resized), cv2.COLORMAP_JET)
    gallery_heatmap = cv2.cvtColor(gallery_heatmap, cv2.COLOR_BGR2RGB)

    # Create overlays
    query_overlay = cv2.addWeighted(np.array(query_img), 0.5, query_heatmap, 0.5, 0)
    gallery_overlay = cv2.addWeighted(np.array(gallery_img), 0.5, gallery_heatmap, 0.5, 0)

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Query row
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title('Query Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(query_attn_resized, cmap='jet')
    axes[0, 1].set_title('Query Attention')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(query_overlay)
    axes[0, 2].set_title('Query Overlay')
    axes[0, 2].axis('off')

    # Gallery row
    axes[1, 0].imshow(gallery_img)
    axes[1, 0].set_title('Gallery Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gallery_attn_resized, cmap='jet')
    axes[1, 1].set_title('Gallery Attention')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(gallery_overlay)
    axes[1, 2].set_title('Gallery Overlay')
    axes[1, 2].axis('off')

    # Add similarity score
    fig.suptitle(f'Similarity Score: {similarity:.4f}', fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'comparison_with_attention.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Similarity between query and gallery: {similarity:.4f}")
    print(f"Saved comparison to {output_path}")

    return similarity


def main():
    parser = argparse.ArgumentParser(description='PromptSG Inference with GradCAM')

    parser.add_argument('--config', type=str, default='configs/person/vit_promptsg.yml',
                        help='Path to config file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for visualization')
    parser.add_argument('--query', type=str, default=None,
                        help='Path to query image for comparison')
    parser.add_argument('--gallery', type=str, default=None,
                        help='Path to gallery image for comparison')
    parser.add_argument('--output_dir', type=str, default='./gradcam_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Load config
    cfg.merge_from_file(args.config)
    cfg.freeze()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model (we need num_classes, camera_num, view_num)
    # For inference, we can use dummy values since we're only using features
    print("Loading model...")

    # Try to load from dataloader to get correct num_classes
    try:
        from datasets.make_dataloader_promptsg import make_dataloader
        _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    except Exception as e:
        print(f"Could not load dataloader: {e}")
        print("Using default values for num_classes=751, camera_num=6, view_num=1")
        num_classes = 751  # Market-1501 default
        camera_num = 6
        view_num = 1

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # Initialize prompt_composer buffers before loading weights
    # This ensures the buffers have the correct shape to receive checkpoint data
    print("Initializing prompt composer...")
    model.prompt_composer._ensure_embeddings()

    # Load weights
    print(f"Loading weights from {args.weights}...")
    state_dict = torch.load(args.weights, map_location='cpu')

    # Handle DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"Model loaded. Using device: {device}")

    if args.query and args.gallery:
        # Comparison mode
        compare_with_gradcam(
            model=model,
            query_path=args.query,
            gallery_path=args.gallery,
            cfg=cfg,
            output_dir=args.output_dir,
            device=device
        )

    elif args.image:
        # Single image mode
        image_tensor, original_image = load_image(args.image, cfg)

        image_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join(args.output_dir, f'{image_name}_attention.png')

        visualize_attention(
            model=model,
            image_tensor=image_tensor,
            original_image=original_image,
            output_path=output_path,
            device=device
        )

    else:
        print("Please provide either --image or both --query and --gallery")


if __name__ == '__main__':
    main()
