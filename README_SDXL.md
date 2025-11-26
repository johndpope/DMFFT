# DMFFT for SDXL

This document explains the SDXL-specific adaptations of the DMFFT (Diffusion Model Fourier Feature Transform) technique.

## SDXL vs SD 2.1 Architecture Comparison

### U-Net Decoder Structure

| Block Index | SD 2.1 | SDXL | Channels |
|-------------|--------|------|----------|
| up_blocks[0] | UpBlock2D | UpBlock2D | 1280 |
| up_blocks[1] | CrossAttnUpBlock2D | CrossAttnUpBlock2D | 1280 |
| up_blocks[2] | CrossAttnUpBlock2D | CrossAttnUpBlock2D | 640 |
| up_blocks[3] | CrossAttnUpBlock2D | N/A | 320 (SD only) |

**Key Difference:** SDXL has 3 up_blocks vs 4 in SD 2.1, and doesn't have 320-channel blocks.

### Attention Mechanism

| Aspect | SD 2.1 | SDXL |
|--------|--------|------|
| Cross-Attention | CrossAttention | Transformer2DModel |
| Text Encoder | CLIP ViT-H | CLIP ViT-L + OpenCLIP ViT-bigG |
| Resolution | 768x768 | 1024x1024 |

## File Structure

```
DMFFT/
├── app.py              # Original SD 2.1 Gradio interface (with new documentation)
├── app_sdxl.py         # NEW: SDXL Gradio interface
├── utils.py            # Original SD 2.1 utilities
├── utils_sdxl.py       # NEW: SDXL-compatible utilities with detailed docs
└── README_SDXL.md      # This file
```

## Quick Start (SDXL)

```bash
# Install dependencies
pip install diffusers transformers accelerate gradio torch

# Run SDXL version
python app_sdxl.py
```

## Parameter Quick Reference

### Most Important Parameters

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `b1, b2` | Backbone LF scale (semantics) | 1.2-1.4 |
| `s1, s2` | Backbone HF scale (sharpness) | 1.2-1.4 |
| `b1_1, b2_1` | Skip LF scale (clarity) | 0.0-0.8 |
| `types` | Processing mode | 4 (recommended) |
| `blend` | Where to apply a/p | 2 (HF only) |

### Parameter Naming Convention

```
  k1    = Channel fraction for 1280ch BACKBONE
  k1_1  = Channel fraction for 1280ch SKIP
  k2    = Channel fraction for 640ch BACKBONE
  k2_1  = Channel fraction for 640ch SKIP

  Similar pattern for b, s, t, g, a, p parameters
```

## Recommended Settings

### General Quality Enhancement

```python
register_tune_crossattn_upblock2d_sdxl(
    pipe,
    types=4,              # Full DMFFT mode

    # Channel selection (which channels to modify)
    k1=1.0, k2=1.0,       # All backbone channels
    k1_1=1.0, k2_1=1.0,   # All skip channels

    # Backbone LF/HF scaling (semantics + sharpness)
    b1=1.3, b2=1.3,       # Enhance low-frequency structure
    s1=1.3, s2=1.3,       # Enhance high-frequency details

    # Skip LF/HF scaling (color/texture control)
    b1_1=0.4, b2_1=0.4,   # Reduce skip LF for clarity
    s1_1=1.0, s2_1=1.0,   # Preserve skip HF details

    # Frequency threshold
    t1=2, t2=2,           # Size of low-frequency region
    t1_1=2, t2_1=2,

    # Blend type (2 = apply only to high frequencies)
    blend1=2, blend2=2,
    blend1_1=2, blend2_1=2,

    # Iteration control
    skips=0,              # Start from first ResNet block
    tunes=3,              # Tune all 3 ResNet blocks
)
```

### Maximum Sharpness

```python
s1=1.5, s2=1.5          # Increase HF scaling
```

### Cleaner Colors

```python
b1_1=0.2, b2_1=0.2      # Further reduce skip LF
```

## Research Paper Summary

From the DMFFT paper:

1. **CrossAttnUpBlock is KEY** - This is where semantic control happens
2. **Backbone controls semantics** - The hidden_states path affects what objects look like
3. **Skip controls color/texture** - The res_hidden_states path affects appearance details
4. **High-freq scaling [1.2-1.4]** - Improves both semantics and color
5. **Low-freq scaling on skip [0.0-0.8]** - Dramatically improves clarity

## Troubleshooting

### Out of Memory

SDXL requires ~8-10GB VRAM. Try:
```python
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
```

### No Visual Difference

- Ensure `types=4` (not 0)
- Ensure `tunes >= 1`
- Try more extreme values: `b1=1.5, s1=1.5`

### Artifacts

- Reduce HF scaling: `s1=1.1, s2=1.1`
- Keep phase at 1.0: `p1=1.0, p2=1.0`

## Technical Details

### How DMFFT Works

1. **Intercept Features**: Hook into U-Net up_blocks
2. **FFT Transform**: Convert spatial features to frequency domain
3. **Scale Frequencies**: Boost/reduce low/high frequency components
4. **IFFT Transform**: Convert back to spatial domain
5. **Continue Diffusion**: Modified features flow through rest of U-Net

### Why CrossAttnUpBlock Matters More

CrossAttnUpBlock combines:
- ResNet blocks (feature transformation)
- Transformer blocks (text-to-image attention)

The attention mechanism integrates text understanding, so modifying features HERE affects semantic interpretation more than in UpBlock2D.
