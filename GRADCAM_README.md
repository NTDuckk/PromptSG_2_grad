# GradCAM Visualization for PromptSG

This document explains how to use the GradCAM (Gradient-weighted Class Activation Mapping) visualization features integrated into the PromptSG model.

## Overview

GradCAM provides visual explanations for the model's predictions by highlighting important regions in the input image. The PromptSG implementation includes:

1. **Standard GradCAM** - Gradient-based attention visualization
2. **GradCAM++** - Improved localization with better weighting
3. **MIM Attention** - Direct visualization of the Multimodal Interaction Module's cross-attention

## Files Added

```
PromptSG_2/
├── utils/
│   └── gradcam.py              # Core GradCAM implementation
├── visualize_gradcam.py         # Full visualization script
├── inference_with_gradcam.py    # Simplified inference with attention
└── GRADCAM_README.md            # This documentation
```

## Quick Start

### 1. Basic Attention Visualization During Testing

```bash
python test_promptsg.py \
    --config_file configs/person/vit_promptsg.yml \
    --visualize \
    --num_visualize 50 \
    TEST.WEIGHT path/to/model.pth \
    OUTPUT_DIR ./test_output
```

This will:
- Run standard evaluation (mAP, Rank-1, etc.)
- Save attention visualizations for the first 50 images
- Output saved to `./test_output/attention_visualizations/`

### 2. Single Image Visualization

```bash
python inference_with_gradcam.py \
    --config configs/person/vit_promptsg.yml \
    --weights path/to/model.pth \
    --image path/to/query_image.jpg \
    --output_dir ./gradcam_results
```

### 3. Compare Query and Gallery Images

```bash
python inference_with_gradcam.py \
    --config configs/person/vit_promptsg.yml \
    --weights path/to/model.pth \
    --query path/to/query.jpg \
    --gallery path/to/gallery.jpg \
    --output_dir ./gradcam_results
```

This outputs:
- Side-by-side attention maps
- Computed similarity score between images

### 4. Full GradCAM Visualization Script

For more control and multiple visualization methods:

```bash
python visualize_gradcam.py \
    --config_file configs/person/vit_promptsg.yml \
    --weight path/to/model.pth \
    --image_path path/to/image.jpg \
    --method all \
    --output_dir ./gradcam_outputs
```

Options for `--method`:
- `gradcam` - Standard GradCAM
- `gradcam++` - GradCAM++ (better localization)
- `mim_attention` - Direct MIM cross-attention visualization
- `all` - Generate all visualizations

Options for `--layer_type`:
- `mim` - Multimodal Interaction Module (default)
- `vit_last` - Last ViT transformer block
- `vit_proj` - ViT projection layer
- `classifier` - Classification head
- `bottleneck` - Bottleneck layer

### 5. Batch Visualization

```bash
python visualize_gradcam.py \
    --config_file configs/person/vit_promptsg.yml \
    --weight path/to/model.pth \
    --batch_mode \
    --num_samples 100 \
    --method mim_attention \
    --output_dir ./batch_gradcam
```

## Programmatic Usage

### Using the Model's Built-in Attention Method

```python
from model.make_model_promptsg import make_model
from config import cfg

# Load model
cfg.merge_from_file('configs/person/vit_promptsg.yml')
model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
model.load_param('path/to/weights.pth')
model.eval()

# Get attention map
image_tensor = ...  # Your preprocessed image (B, C, H, W)
attn_map = model.get_attention_map(image_tensor, reshape_to_image=True)
# attn_map shape: (B, H_patches, W_patches)
```

### Using the Forward with Attention Method

```python
# Get detailed outputs including attention
result = model.forward_with_attention(image_tensor)

print(result.keys())
# ['logits', 'features', 'mim_attention', 'patch_tokens',
#  'text_features', 'cls_states', 'v_final']

# MIM attention map
attn_map = result['mim_attention']  # (B, 1, num_patches)

# CLS token states from each MIM block
cls_states = result['cls_states']  # List of (B, D) tensors
```

### Using GradCAM Directly

```python
from utils.gradcam import GradCAM, MIMAttentionCAM, visualize_gradcam

# Method 1: MIM Attention (fastest, no gradients needed)
mim_cam = MIMAttentionCAM(model)
heatmap = mim_cam(image_tensor)
mim_cam.remove_hooks()

# Method 2: GradCAM on specific layers
target_layers = [model.mim.cross_attn]
gradcam = GradCAM(model, target_layers)
heatmap = gradcam(image_tensor, target_category=predicted_class)
gradcam.remove_hooks()

# Method 3: Quick visualization
visualize_gradcam(
    model=model,
    image_tensor=image_tensor,
    target_layers=target_layers,
    method='gradcam',
    save_path='output.png',
    show=True
)
```

## Understanding the Visualizations

### MIM Attention Map
The Multimodal Interaction Module (MIM) computes cross-attention between:
- **Query**: Text features from the prompt ("A photo of a X person")
- **Key/Value**: Image patch tokens from ViT

The attention map shows which image regions are most relevant to the text-guided representation. High attention areas indicate parts the model considers most important for person identification.

### GradCAM
GradCAM uses gradients flowing back from the classification output to weight the importance of different feature map locations. It provides a class-discriminative localization.

### GradCAM++
An improvement over GradCAM that uses second-order gradients for better weighting, particularly useful for:
- Multiple instances of the same object
- Better full-body localization

## Output Examples

The visualization scripts generate:

1. **Original Image** - Input image
2. **Heatmap** - Color-coded attention/activation map (red = high, blue = low)
3. **Overlay** - Heatmap overlaid on the original image

For comparison mode:
- Side-by-side query and gallery visualizations
- Computed similarity score

## Tips

1. **Best for ReID Analysis**: Use `mim_attention` to understand what body parts the model focuses on for identification.

2. **Debugging Mismatches**: When the model fails to match persons correctly, compare attention maps to see if it's focusing on wrong regions (e.g., background instead of person).

3. **Model Interpretability**: Use GradCAM to verify the model learns meaningful features (e.g., focusing on clothing, body shape) rather than spurious correlations.

4. **Quick Testing**: The `inference_with_gradcam.py` script is fastest for testing individual images.

## Requirements

The GradCAM functionality requires:
- OpenCV (`pip install opencv-python`)
- Matplotlib (`pip install matplotlib`)
- NumPy (already required by PyTorch)

These should already be installed as part of the PromptSG dependencies.
