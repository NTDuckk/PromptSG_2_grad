"""
GradCAM and GradCAM++ implementation for PromptSG model.

This module provides gradient-weighted class activation mapping for:
1. Vision Transformer attention visualization
2. Multimodal Interaction Module (MIM) cross-attention visualization
3. Classification-based GradCAM

Reference:
- GradCAM: https://arxiv.org/abs/1610.02391
- GradCAM++: https://arxiv.org/abs/1710.11063
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Union
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    """
    GradCAM implementation for PromptSG model.

    Supports multiple target layers:
    - ViT patch embeddings
    - MIM cross-attention
    - Classifier layers
    """

    def __init__(self, model: nn.Module, target_layers: List[nn.Module], use_cuda: bool = True):
        """
        Args:
            model: The PromptSG model
            target_layers: List of layers to compute GradCAM for
            use_cuda: Whether to use CUDA
        """
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.gradients = []
        self.activations = []
        self.handles = []

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layers."""
        for layer in self.target_layers:
            # Forward hook to capture activations
            handle_forward = layer.register_forward_hook(self._save_activation)
            # Backward hook to capture gradients
            handle_backward = layer.register_full_backward_hook(self._save_gradient)

            self.handles.append(handle_forward)
            self.handles.append(handle_backward)

    def _save_activation(self, module, input, output):
        """Save activation during forward pass."""
        if isinstance(output, tuple):
            output = output[0]
        self.activations.append(output.detach())

    def _save_gradient(self, module, grad_input, grad_output):
        """Save gradient during backward pass."""
        if isinstance(grad_output, tuple):
            grad = grad_output[0]
        else:
            grad = grad_output
        self.gradients.append(grad.detach())

    def _clear(self):
        """Clear stored activations and gradients."""
        self.gradients = []
        self.activations = []

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_category: Optional[int] = None,
        eigen_smooth: bool = False
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_category: Target class index. If None, uses predicted class.
            eigen_smooth: Whether to use eigen smoothing

        Returns:
            GradCAM heatmap as numpy array
        """
        self._clear()

        if self.use_cuda:
            input_tensor = input_tensor.cuda()

        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Handle different output formats from PromptSG
        if isinstance(output, (list, tuple)):
            # During training: [cls_score, cls_score_proj], triplet_feats, image_feat, text_feat
            if len(output) == 4:
                cls_scores = output[0]
                if isinstance(cls_scores, list):
                    logits = cls_scores[0]
                else:
                    logits = cls_scores
            else:
                # During inference: concatenated features
                logits = output
        else:
            logits = output

        # If output is features (inference mode), we can't compute class-based GradCAM
        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) > 1000):
            # Likely feature vector, use sum as target
            target = logits.sum()
        else:
            if target_category is None:
                argmax_result = logits.argmax(dim=1)
                target_category = argmax_result.item() if hasattr(argmax_result, 'item') else int(argmax_result)
            target = logits[0, target_category]

        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Compute GradCAM
        gradcam_maps = []

        for activation, gradient in zip(self.activations, reversed(self.gradients)):
            # Global average pooling of gradients
            if activation.dim() == 3:
                # (B, N, D) format for ViT
                weights = gradient.mean(dim=1, keepdim=True)  # (B, 1, D)
                cam = (weights * activation).sum(dim=-1)  # (B, N)
            elif activation.dim() == 4:
                # (B, C, H, W) format for CNN
                weights = gradient.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
                cam = (weights * activation).sum(dim=1)  # (B, H, W)
            else:
                continue

            # ReLU on CAM
            cam = F.relu(cam)

            gradcam_maps.append(cam)

        if not gradcam_maps:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # Use the first (or combine) GradCAM maps
        cam = gradcam_maps[0]

        # Reshape for ViT (if needed)
        if cam.dim() == 2:
            B, N = cam.shape
            # Assume square grid (excluding CLS token if present)
            if hasattr(self.model, 'h_resolution') and hasattr(self.model, 'w_resolution'):
                h, w = self.model.h_resolution, self.model.w_resolution
            else:
                # Estimate grid size
                grid_size = int(np.sqrt(N))
                if grid_size * grid_size == N:
                    h = w = grid_size
                else:
                    # May have CLS token
                    grid_size = int(np.sqrt(N - 1))
                    h = w = grid_size
                    cam = cam[:, 1:]  # Remove CLS token

            cam = cam.view(B, h, w)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        return cam.cpu().numpy()


class GradCAMPlusPlus(GradCAM):
    """
    GradCAM++ implementation with improved weighting.
    """

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_category: Optional[int] = None,
        eigen_smooth: bool = False
    ) -> np.ndarray:
        """Generate GradCAM++ heatmap."""
        self._clear()

        if self.use_cuda:
            input_tensor = input_tensor.cuda()

        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Handle output format
        if isinstance(output, (list, tuple)) and len(output) == 4:
            cls_scores = output[0]
            if isinstance(cls_scores, list):
                logits = cls_scores[0]
            else:
                logits = cls_scores
        else:
            logits = output if not isinstance(output, tuple) else output[0]

        if target_category is None and logits.dim() == 2:
            argmax_result = logits.argmax(dim=1)
            target_category = argmax_result.item() if hasattr(argmax_result, 'item') else int(argmax_result)

        if logits.dim() == 2 and logits.size(1) <= 1000:
            target = logits[0, target_category]
        else:
            target = logits.sum()

        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Compute GradCAM++
        gradcam_maps = []

        for activation, gradient in zip(self.activations, reversed(self.gradients)):
            if activation.dim() == 3:
                # ViT format
                grad_2 = gradient ** 2
                grad_3 = gradient ** 3

                sum_activations = activation.sum(dim=1, keepdim=True)

                alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
                alpha = grad_2 / alpha_denom

                weights = (alpha * F.relu(gradient)).sum(dim=1, keepdim=True)
                cam = (weights * activation).sum(dim=-1)

            elif activation.dim() == 4:
                # CNN format
                grad_2 = gradient ** 2
                grad_3 = gradient ** 3

                sum_activations = activation.sum(dim=(2, 3), keepdim=True)

                alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
                alpha = grad_2 / alpha_denom

                weights = (alpha * F.relu(gradient)).sum(dim=(2, 3), keepdim=True)
                cam = (weights * activation).sum(dim=1)
            else:
                continue

            cam = F.relu(cam)
            gradcam_maps.append(cam)

        if not gradcam_maps:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        cam = gradcam_maps[0]

        # Reshape for ViT
        if cam.dim() == 2:
            B, N = cam.shape
            grid_size = int(np.sqrt(N - 1)) if N > 1 else 1
            if grid_size * grid_size == N - 1:
                cam = cam[:, 1:]
                h = w = grid_size
            else:
                h = w = int(np.sqrt(N))
            cam = cam.view(B, h, w)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        return cam.cpu().numpy()


class MIMAttentionCAM:
    """
    Specialized visualization for Multimodal Interaction Module attention.

    This extracts and visualizes the cross-attention weights between
    text features and image patches directly from the MIM module.
    """

    def __init__(self, model: nn.Module, use_cuda: bool = True):
        """
        Args:
            model: The PromptSG model
            use_cuda: Whether to use CUDA
        """
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.attention_maps = []
        self.handles = []

        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on MIM cross-attention."""
        # Find MIM module
        mim_module = None
        for name, module in self.model.named_modules():
            if 'mim' in name.lower() and hasattr(module, 'cross_attn'):
                mim_module = module
                break

        if mim_module is None:
            print("Warning: Could not find MIM module")
            return

        # Hook on cross_attn
        def hook_fn(module, input, output):
            # output is (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) > 1:
                self.attention_maps.append(output[1].detach())

        handle = mim_module.cross_attn.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def remove_hooks(self):
        """Remove hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract MIM attention map.

        Args:
            input_tensor: Input image tensor

        Returns:
            Attention map as numpy array
        """
        self.attention_maps = []

        if self.use_cuda:
            input_tensor = input_tensor.cuda()

        self.model.eval()

        with torch.no_grad():
            _ = self.model(input_tensor)

        if not self.attention_maps:
            print("No attention maps captured")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # Get attention map (B, num_heads, 1, num_patches)
        attn = self.attention_maps[0]

        # Average over heads
        attn = attn.mean(dim=1)  # (B, 1, num_patches)
        attn = attn.squeeze(1)   # (B, num_patches)

        # Reshape to spatial
        B, N = attn.shape

        if hasattr(self.model, 'h_resolution') and hasattr(self.model, 'w_resolution'):
            h, w = self.model.h_resolution, self.model.w_resolution
        else:
            h = w = int(np.sqrt(N))

        attn = attn.view(B, h, w)

        # Normalize
        attn = attn - attn.min()
        if attn.max() > 0:
            attn = attn / attn.max()

        # Resize to input size
        attn = F.interpolate(
            attn.unsqueeze(1),
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        return attn.cpu().numpy()


def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Apply colormap to heatmap.

    Args:
        heatmap: Grayscale heatmap (H, W) with values in [0, 1]
        colormap: OpenCV colormap

    Returns:
        Colored heatmap as (H, W, 3) uint8 array
    """
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image.

    Args:
        image: Original image (H, W, 3) uint8
        heatmap: Grayscale heatmap (H, W) with values in [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap

    Returns:
        Blended image as (H, W, 3) uint8 array
    """
    colored_heatmap = apply_colormap(heatmap, colormap)

    # Resize heatmap to image size if needed
    if colored_heatmap.shape[:2] != image.shape[:2]:
        colored_heatmap = cv2.resize(colored_heatmap, (image.shape[1], image.shape[0]))

    # Blend
    output = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)

    return output


def denormalize_image(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Denormalize image tensor to numpy array.

    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized image as (H, W, 3) uint8 array
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.cpu().clone()

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    tensor = tensor.clamp(0, 1)

    # Convert to numpy
    image = tensor.permute(1, 2, 0).numpy()
    image = np.uint8(255 * image)

    return image


def visualize_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layers: List[nn.Module],
    target_category: Optional[int] = None,
    method: str = 'gradcam',
    save_path: Optional[str] = None,
    show: bool = True
) -> Dict[str, np.ndarray]:
    """
    Visualize GradCAM for a given image.

    Args:
        model: The PromptSG model
        image_tensor: Input image tensor (B, C, H, W)
        target_layers: Layers to compute GradCAM for
        target_category: Target class (None for predicted)
        method: 'gradcam', 'gradcam++', or 'mim_attention'
        save_path: Path to save visualization
        show: Whether to display the plot

    Returns:
        Dictionary containing heatmaps and overlays
    """
    # Select method
    if method == 'gradcam':
        cam = GradCAM(model, target_layers)
    elif method == 'gradcam++':
        cam = GradCAMPlusPlus(model, target_layers)
    elif method == 'mim_attention':
        cam = MIMAttentionCAM(model)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Generate heatmap
    heatmap = cam(image_tensor, target_category)

    # Clean up
    if hasattr(cam, 'remove_hooks'):
        cam.remove_hooks()

    # Denormalize image
    original_image = denormalize_image(image_tensor)

    # Create overlay
    overlay = overlay_heatmap(original_image, heatmap)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(f'{method.upper()} Heatmap')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return {
        'heatmap': heatmap,
        'overlay': overlay,
        'original': original_image
    }


def batch_visualize_gradcam(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    target_layers: List[nn.Module],
    num_samples: int = 10,
    method: str = 'gradcam',
    output_dir: str = './gradcam_outputs',
    device: str = 'cuda'
) -> None:
    """
    Batch visualization of GradCAM for multiple images.

    Args:
        model: The PromptSG model
        dataloader: DataLoader with images
        target_layers: Layers to compute GradCAM for
        num_samples: Number of samples to visualize
        method: GradCAM method
        output_dir: Output directory
        device: Device to use
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    model.to(device)
    model.eval()

    count = 0
    for batch_idx, batch in enumerate(dataloader):
        if count >= num_samples:
            break

        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            images = batch[0]
            if len(batch) > 1:
                pids = batch[1]
            else:
                pids = None
        else:
            images = batch
            pids = None

        for i in range(images.size(0)):
            if count >= num_samples:
                break

            image = images[i:i+1].to(device)

            pid_val = pids[i].item() if hasattr(pids[i], 'item') else pids[i]
            pid_str = f"_pid{pid_val}" if pids is not None else ""
            save_path = os.path.join(output_dir, f'gradcam_{count}{pid_str}.png')

            try:
                visualize_gradcam(
                    model=model,
                    image_tensor=image,
                    target_layers=target_layers,
                    method=method,
                    save_path=save_path,
                    show=False
                )
                print(f"Processed image {count + 1}/{num_samples}")
            except Exception as e:
                print(f"Error processing image {count}: {e}")

            count += 1

    print(f"Saved {count} GradCAM visualizations to {output_dir}")
