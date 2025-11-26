"""
DMFFT (Diffusion Model Fourier Feature Transform) - SDXL Version
================================================================

This module provides Fourier-domain feature manipulation for SDXL (Stable Diffusion XL).
Based on the DMFFT research paper, it modifies U-Net upsampling blocks to enhance
image generation quality through frequency-domain scaling.

SDXL U-Net Architecture (relevant for DMFFT):
----------------------------------------------
SDXL uses a modified U-Net with the following upsampling structure:

up_blocks[0]: UpBlock2D (1280 channels) - Pure upsampling, no cross-attention
              - Contains 3 ResNet blocks
              - Handles high-level semantic features

up_blocks[1]: CrossAttnUpBlock2D (1280 channels) - MOST IMPORTANT FOR EDITS
              - Contains 3 ResNet blocks + 3 Transformer2DModel blocks
              - This is where semantic content is primarily controlled
              - Channel dimension: 1280

up_blocks[2]: CrossAttnUpBlock2D (640 channels)
              - Contains 3 ResNet blocks + 3 Transformer2DModel blocks
              - Handles mid-level features (shapes, colors)
              - Channel dimension: 640

Comparison with SD 1.x/2.x:
---------------------------
SD 2.1:  1280 -> 1280 -> 640 -> 320 channels in decoder
SDXL:    1280 -> 1280 -> 640 channels in decoder (no 320 channel blocks)

Key Insight from Paper:
-----------------------
"CrossAttnUpBlock is the key block to modify (not UpBlock)"
- Backbone features (hidden_states) affect SEMANTICS
- Skip features (res_hidden_states) affect COLOR and TEXTURE

Parameter Reference Guide:
==========================

CHANNEL SELECTION (k parameters):
---------------------------------
k1, k2:      Fraction of backbone channels to modify (hidden_states)
             - k=0.0: No channels modified
             - k=0.5: First 50% of channels modified
             - k=1.0: All channels modified (100%)
             - Higher k = more global changes to semantics/structure

k1_1, k2_1:  Fraction of skip connection channels to modify (res_hidden_states)
             - Controls how much skip connection info is modified
             - Skip connections carry fine details from encoder

Suffix convention:
  - "1" suffix: Applies to 1280-channel blocks (higher-level features)
  - "2" suffix: Applies to 640-channel blocks (mid-level features)
  - "_1" suffix: Applies to skip connections (res_hidden_states)
  - No "_1" suffix: Applies to backbone (hidden_states)

BACKBONE/LOW-FREQUENCY SCALING (b parameters):
----------------------------------------------
b1:          Backbone LF scale for 1280-ch blocks
             - Recommended: [1.2-1.4] for enhanced semantics
             - >1.0: Amplifies low-frequency (coarse structure) in backbone
             - <1.0: Reduces coarse structure strength

b2:          Backbone LF scale for 640-ch blocks
             - Recommended: [1.2-1.4] for better color consistency
             - Affects shape and color distribution

b1_1, b2_1:  Skip connection LF scale
             - Recommended: [0.0-0.8] for improved clarity
             - Lower values = cleaner, less blurry results

SKIP/HIGH-FREQUENCY SCALING (s parameters):
-------------------------------------------
s1:          Backbone HF scale for 1280-ch blocks
             - Recommended: [1.2-1.4] for enhanced details
             - >1.0: Sharpens high-frequency details

s2:          Backbone HF scale for 640-ch blocks
             - Affects texture and edge sharpness

s1_1, s2_1:  Skip connection HF scale
             - Control fine detail preservation from encoder
             - Higher = more detail from original structure

FREQUENCY THRESHOLD (t parameters):
----------------------------------
t1, t2:      Frequency cutoff threshold (in pixels from center)
             - Determines boundary between low/high frequency
             - t=1: Very small LF region (mostly HF manipulation)
             - t=10: Larger LF region
             - Recommended: 1-5 for most applications

t1_1, t2_1:  Threshold for skip connections

GLOBAL SCALING (g parameters):
------------------------------
g1, g2:      Global scale multiplier for backbone features
             - Applied BEFORE frequency decomposition
             - g=1.0: No global change
             - g>1.0: Overall feature amplification
             - g<1.0: Overall feature reduction

g1_1, g2_1:  Global scale for skip connections

AMPLITUDE SCALING (a parameters):
---------------------------------
a1, a2:      Amplitude scaling in Fourier domain (backbone)
             - Affects the magnitude of frequency components
             - a>1.0: Increases contrast/intensity
             - a<1.0: Reduces contrast

a1_1, a2_1:  Amplitude scaling for skip connections

PHASE SCALING (p parameters):
-----------------------------
p1, p2:      Phase scaling in Fourier domain (backbone)
             - Phase encodes spatial position/structure information
             - p=1.0: Preserve original phase
             - p!=1.0: May cause spatial distortions (use carefully)

p1_1, p2_1:  Phase scaling for skip connections

BLEND TYPES:
------------
blend_type controls how amplitude/phase scaling is applied:

-1: No FFT applied (only global_scale used)
 0: Apply a/p scaling to ENTIRE frequency spectrum
 1: Apply a/p scaling only to LOW frequencies (inside threshold)
 2: Apply a/p scaling only to HIGH frequencies (outside threshold)
 3: LF amplitude + HF phase scaling
 4: HF amplitude + LF phase scaling

Recommended: blend_type=2 (scale high frequencies only)

ITERATION CONTROL:
------------------
skips:       Number of ResNet blocks to skip before applying FFT
             - skips=0: Apply from first block
             - Higher = apply only to later blocks in decoder

tunes:       Number of ResNet blocks to tune (after skips)
             - tunes=3: Tune 3 blocks
             - Set to number of ResNet blocks in target up_block

TYPE MODES:
-----------
types=0: No modification (baseline)
types=1: Simple channel scaling + fourier_filter on skip (legacy mode)
types=2: Direct channel scaling (no FFT)
types=3: Adaptive scaling based on feature statistics (experimental)
types=4: Full fourier_solo on selected blocks (RECOMMENDED for DMFFT)
types=5: Channel-dimension-aware fourier_solo (separate 1280/640 handling)

RECOMMENDED SETTINGS FOR SDXL:
==============================
For enhanced image quality:
  types=4, blend=2
  k1=0.5, k2=0.5, k1_1=1.0, k2_1=1.0
  b1=1.3, b2=1.3, b1_1=0.4, b2_1=0.4  (LF: enhance backbone, reduce skip)
  s1=1.3, s2=1.3, s1_1=1.0, s2_1=1.0  (HF: enhance backbone, preserve skip)
  t1=2, t2=2, t1_1=2, t2_1=2
  a1=1.0, a2=1.0, a1_1=1.0, a2_1=1.0
  p1=1.0, p2=1.0, p1_1=1.0, p2_1=1.0
  g1=1.0, g2=1.0, g1_1=1.0, g2_1=1.0
  skips=0, tunes=3

For sharper details:
  Increase s1, s2 to 1.4-1.5

For better color consistency:
  Increase b1, b2 to 1.4
  Decrease b1_1, b2_1 to 0.2-0.3
"""

import numpy
import torch
import torch.fft as fft
from diffusers.utils import is_torch_version
from typing import Any, Dict, Optional, Tuple

# Debug integration - optional, only used when debug is enabled
_debug_capture_enabled = False
_debug_capture_callback = None


def set_debug_capture(enabled: bool, callback=None):
    """
    Enable/disable feature capture for debugging.

    Args:
        enabled: Whether to capture features
        callback: Function(name, hidden_states, res_hidden_states, block_idx) to call
    """
    global _debug_capture_enabled, _debug_capture_callback
    _debug_capture_enabled = enabled
    _debug_capture_callback = callback


def _capture_features(name: str, hidden_states: torch.Tensor,
                      res_hidden_states: Optional[torch.Tensor], block_idx: int):
    """Internal function to capture features if debug is enabled."""
    if _debug_capture_enabled and _debug_capture_callback is not None:
        _debug_capture_callback(name, hidden_states, res_hidden_states, block_idx)


def isinstance_str(x: object, cls_name: str) -> bool:
    """Check if object is instance of class by name (handles dynamic class loading)."""
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


def fourier_filter(x: torch.Tensor, threshold: int, scale: float) -> torch.Tensor:
    """
    Apply Fourier low-pass filter with scaling.

    Args:
        x: Input tensor [B, C, H, W]
        threshold: Radius of low-frequency region (pixels from center)
        scale: Multiplier for low-frequency components

    Returns:
        Filtered tensor with scaled low frequencies
    """
    dtype = x.dtype
    device = x.device
    x = x.type(torch.float32)

    # Transform to frequency domain
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))  # Center low frequencies

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=device)

    # Define low-frequency region (center of spectrum)
    crow, ccol = H // 2, W // 2
    top = max(0, crow - threshold)
    left = max(0, ccol - threshold)

    # Scale low-frequency region
    mask[..., top:crow + threshold, left:ccol + threshold] = scale
    x_freq = x_freq * mask

    # Transform back to spatial domain
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.type(dtype)


def fourier_solo(
    x: torch.Tensor,
    global_scale: float = 1.0,
    freq_threshold: int = 0,
    lf_scale: float = 1.0,
    hf_scale: float = 1.0,
    amplitude_scale: float = 1.0,
    phase_scale: float = 1.0,
    blend_type: int = 0
) -> torch.Tensor:
    """
    Advanced Fourier-domain feature manipulation with separate amplitude/phase control.

    This is the core DMFFT function that allows fine-grained control over:
    - Low vs high frequency scaling
    - Amplitude (intensity) manipulation
    - Phase (structure) manipulation

    Args:
        x: Input tensor [B, C, H, W] - feature maps from U-Net
        global_scale: Pre-FFT multiplier (g parameter)
        freq_threshold: Radius defining low-frequency region (t parameter)
        lf_scale: Low-frequency scaling factor (b parameter)
        hf_scale: High-frequency scaling factor (s parameter)
        amplitude_scale: Fourier amplitude multiplier (a parameter)
        phase_scale: Fourier phase multiplier (p parameter)
        blend_type: How to apply amplitude/phase scaling (-1 to 4)

    Returns:
        Modified tensor with frequency-domain adjustments applied

    Blend Types Explained:
        -1: Skip FFT entirely, only apply global_scale
         0: Apply amplitude/phase scaling to ALL frequencies
         1: Apply amplitude/phase scaling only to LOW frequencies
         2: Apply amplitude/phase scaling only to HIGH frequencies (recommended)
         3: LOW freq amplitude + HIGH freq phase scaling
         4: HIGH freq amplitude + LOW freq phase scaling
    """
    dtype = x.dtype
    device = x.device
    x = x.type(torch.float32)

    # Apply global scaling before FFT
    x = x * global_scale

    # Early exit if no FFT needed
    if blend_type == -1:
        return x.type(dtype)

    # Transform to frequency domain
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape

    # Create frequency mask: hf_scale everywhere, lf_scale in center
    mask = torch.ones((B, C, H, W), device=device) * hf_scale

    crow, ccol = H // 2, W // 2
    top = max(0, crow - freq_threshold)
    left = max(0, ccol - freq_threshold)

    # Set low-frequency region
    mask[..., top:crow + freq_threshold, left:ccol + freq_threshold] = lf_scale
    x_freq = x_freq * mask

    # Decompose into amplitude and phase
    amplitude = torch.abs(x_freq)
    phase = torch.angle(x_freq)

    # Apply amplitude/phase scaling based on blend_type
    if blend_type == 0:
        # Scale ALL frequencies
        amplitude = amplitude * amplitude_scale
        phase = phase * phase_scale

    elif blend_type == 1:
        # Scale only LOW frequencies
        amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold] *= amplitude_scale
        phase[..., top:crow + freq_threshold, left:ccol + freq_threshold] *= phase_scale

    elif blend_type == 2:
        # Scale only HIGH frequencies (preserve LF amplitude/phase)
        lf_amplitude = amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold].clone()
        lf_phase = phase[..., top:crow + freq_threshold, left:ccol + freq_threshold].clone()

        amplitude = amplitude * amplitude_scale
        phase = phase * phase_scale

        # Restore low-frequency components
        amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold] = lf_amplitude
        phase[..., top:crow + freq_threshold, left:ccol + freq_threshold] = lf_phase

    elif blend_type == 3:
        # LF amplitude + HF phase
        amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold] *= amplitude_scale

        lf_phase = phase[..., top:crow + freq_threshold, left:ccol + freq_threshold].clone()
        phase = phase * phase_scale
        phase[..., top:crow + freq_threshold, left:ccol + freq_threshold] = lf_phase

    elif blend_type == 4:
        # HF amplitude + LF phase
        lf_amplitude = amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold].clone()
        amplitude = amplitude * amplitude_scale
        amplitude[..., top:crow + freq_threshold, left:ccol + freq_threshold] = lf_amplitude

        phase[..., top:crow + freq_threshold, left:ccol + freq_threshold] *= phase_scale

    # Reconstruct complex frequency representation
    x_freq = amplitude * torch.exp(1j * phase)

    # Transform back to spatial domain
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.type(dtype)


def register_tune_upblock2d_sdxl(
    model,
    types: int = 0,
    # 1280-channel block parameters (up_blocks[0] in SDXL - UpBlock2D)
    k1: float = 0.5,      # Backbone channel fraction for 1280ch
    b1: float = 1.2,      # Backbone LF scale [1.2-1.4 recommended]
    t1: int = 1,          # Frequency threshold
    s1: float = 0.9,      # Backbone HF scale
    # Skip connection parameters for 1280ch
    k1_1: float = 0.5,    # Skip channel fraction
    b1_1: float = 1.2,    # Skip LF scale [0.0-0.8 for clarity]
    t1_1: int = 1,        # Skip frequency threshold
    s1_1: float = 0.9,    # Skip HF scale
    # 640-channel block parameters (not present in SDXL UpBlock2D, but kept for compatibility)
    k2: float = 0.5,
    b2: float = 1.4,
    t2: int = 1,
    s2: float = 0.2,
    k2_1: float = 0.5,
    b2_1: float = 1.4,
    t2_1: int = 1,
    s2_1: float = 0.2,
    # Global scaling
    g1: float = 1.0,      # Global backbone scale 1280ch
    g2: float = 1.0,      # Global backbone scale 640ch
    g1_1: float = 1.0,    # Global skip scale 1280ch
    g2_1: float = 1.0,    # Global skip scale 640ch
    # Blend types (0-4, see docstring above)
    blend1: int = 0,
    blend2: int = 0,
    blend1_1: int = 0,
    blend2_1: int = 0,
    # Amplitude scaling
    a1: float = 1.0,
    a2: float = 1.0,
    a1_1: float = 1.0,
    a2_1: float = 1.0,
    # Phase scaling
    p1: float = 1.0,
    p2: float = 1.0,
    p1_1: float = 1.0,
    p2_1: float = 1.0,
    # Iteration control
    skips: int = 0,       # ResNet blocks to skip
    tunes: int = 0        # ResNet blocks to tune
):
    """
    Register DMFFT modifications on UpBlock2D blocks in SDXL U-Net.

    SDXL Architecture Note:
    -----------------------
    In SDXL, up_blocks[0] is an UpBlock2D with 1280 channels.
    This block does NOT have cross-attention, only ResNet blocks.
    It's responsible for high-level feature upsampling.

    The k1/b1/s1 parameters target 1280-channel features.
    The k2/b2/s2 parameters are kept for API compatibility but may not
    be used if no 640-channel UpBlock2D exists.

    Args:
        model: Diffusion pipeline with .unet attribute (SDXL pipeline)
        types: Processing mode (0=off, 4=recommended DMFFT mode)
        k1-p2_1: See module docstring for detailed parameter descriptions
        skips: Number of initial ResNet blocks to skip
        tunes: Number of ResNet blocks to apply DMFFT to
    """
    def up_forward(self):
        def forward(hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, scale=1.0):
            """
            Modified forward pass for UpBlock2D with DMFFT.

            SDXL UpBlock2D contains 3 ResNet blocks that progressively
            upsample features. DMFFT can modify both:
            - hidden_states: Main backbone features (semantic content)
            - res_hidden_states: Skip connections from encoder (details)
            """
            skipping = 0
            finetune = 0

            for resnet_idx, resnet in enumerate(self.resnets):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # Capture features for debugging
                _capture_features("UpBlock2D", hidden_states, res_hidden_states, resnet_idx)

                # Type 1: Simple channel scaling + LF filter on skip
                if self.types == 1:
                    if hidden_states.shape[1] == 1280:
                        # Scale first k1 fraction of backbone channels
                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] *= self.b1
                        # Apply LF filter to skip connection
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t1, scale=self.s1)
                    if hidden_states.shape[1] == 640:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] *= self.b2
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t2, scale=self.s2)

                # Type 2: Direct channel scaling (no FFT)
                elif self.types == 2:
                    if skipping >= self.skips and finetune < self.tunes:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] *= self.b1
                        res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] *= self.s1
                        finetune += 1

                # Type 3: Adaptive scaling based on feature statistics
                elif self.types == 3:
                    hidden_states_numpy = hidden_states.to('cpu').detach().numpy()
                    hidden_states_avg = numpy.mean(hidden_states_numpy, axis=1)
                    hidden_states_max = numpy.max(hidden_states_avg, axis=(1, 2))
                    hidden_states_min = numpy.min(hidden_states_avg, axis=(1, 2))

                    if hidden_states.shape[1] == 1280:
                        hidden_states_relation = numpy.zeros((
                            hidden_states.shape[0],
                            int(hidden_states.shape[1] * self.k1),
                            hidden_states.shape[2],
                            hidden_states.shape[3]
                        ))
                        for index in range(hidden_states.shape[0]):
                            hidden_states_relation[index, :, ...] = (
                                (hidden_states_avg[index, ...] - hidden_states_min[index]) /
                                (hidden_states_max[index] - hidden_states_min[index] + 1e-8)
                            )
                        hidden_states_alpha = 1.0 + (self.b1 - 1.0) * hidden_states_relation
                        hidden_states_alpha = torch.from_numpy(hidden_states_alpha).to(hidden_states.device)
                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] *= hidden_states_alpha
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t1, scale=self.s1)

                    if hidden_states.shape[1] == 640:
                        hidden_states_relation = numpy.zeros((
                            hidden_states.shape[0],
                            int(hidden_states.shape[1] * self.k2),
                            hidden_states.shape[2],
                            hidden_states.shape[3]
                        ))
                        for index in range(hidden_states.shape[0]):
                            hidden_states_relation[index, :, ...] = (
                                (hidden_states_avg[index, ...] - hidden_states_min[index]) /
                                (hidden_states_max[index] - hidden_states_min[index] + 1e-8)
                            )
                        hidden_states_alpha = 1.0 + (self.b2 - 1.0) * hidden_states_relation
                        hidden_states_alpha = torch.from_numpy(hidden_states_alpha).to(hidden_states.device)
                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] *= hidden_states_alpha
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t2, scale=self.s2)

                # Type 4: Full DMFFT with fourier_solo (RECOMMENDED)
                elif self.types == 4:
                    if skipping >= self.skips and finetune < self.tunes:
                        # Apply DMFFT to backbone features
                        if self.k1 > 0.0:
                            hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = fourier_solo(
                                hidden_states[:, :int(hidden_states.shape[1] * self.k1)],
                                global_scale=self.g1,
                                freq_threshold=self.t1,
                                lf_scale=self.b1,
                                hf_scale=self.s1,
                                amplitude_scale=self.a1,
                                phase_scale=self.p1,
                                blend_type=self.blend1
                            )
                        # Apply DMFFT to skip connection
                        if self.k1_1 > 0.0:
                            res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] = fourier_solo(
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)],
                                global_scale=self.g1_1,
                                freq_threshold=self.t1_1,
                                lf_scale=self.b1_1,
                                hf_scale=self.s1_1,
                                amplitude_scale=self.a1_1,
                                phase_scale=self.p1_1,
                                blend_type=self.blend1_1
                            )
                        finetune += 1

                # Type 5: Channel-dimension-aware DMFFT
                elif self.types == 5:
                    if skipping >= self.skips and finetune < self.tunes:
                        if hidden_states.shape[1] == 1280:
                            if self.k1 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = fourier_solo(
                                    hidden_states[:, :int(hidden_states.shape[1] * self.k1)],
                                    global_scale=self.g1, freq_threshold=self.t1,
                                    lf_scale=self.b1, hf_scale=self.s1,
                                    amplitude_scale=self.a1, phase_scale=self.p1,
                                    blend_type=self.blend1
                                )
                            if self.k1_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] = fourier_solo(
                                    res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)],
                                    global_scale=self.g1_1, freq_threshold=self.t1_1,
                                    lf_scale=self.b1_1, hf_scale=self.s1_1,
                                    amplitude_scale=self.a1_1, phase_scale=self.p1_1,
                                    blend_type=self.blend1_1
                                )
                        if hidden_states.shape[1] == 640:
                            if self.k2 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = fourier_solo(
                                    hidden_states[:, :int(hidden_states.shape[1] * self.k2)],
                                    global_scale=self.g2, freq_threshold=self.t2,
                                    lf_scale=self.b2, hf_scale=self.s2,
                                    amplitude_scale=self.a2, phase_scale=self.p2,
                                    blend_type=self.blend2
                                )
                            if self.k2_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] = fourier_solo(
                                    res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)],
                                    global_scale=self.g2_1, freq_threshold=self.t2_1,
                                    lf_scale=self.b2_1, hf_scale=self.s2_1,
                                    amplitude_scale=self.a2_1, phase_scale=self.p2_1,
                                    blend_type=self.blend2_1
                                )
                        finetune += 1

                # Concatenate backbone with skip connection
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                # Process through ResNet block
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

                    if is_torch_version(">=", "1.11.0"):
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                        )
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb
                        )
                else:
                    hidden_states = resnet(hidden_states, temb, scale=scale)

                skipping += 1

            # Final upsampling
            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size, scale=scale)

            return hidden_states

        return forward

    # Apply to all UpBlock2D blocks in the U-Net
    for i, upsample_block in enumerate(model.unet.up_blocks):
        if isinstance_str(upsample_block, "UpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)

            # Store all parameters on the block
            setattr(upsample_block, 'types', types)
            setattr(upsample_block, 'k1', k1)
            setattr(upsample_block, 'b1', b1)
            setattr(upsample_block, 't1', t1)
            setattr(upsample_block, 's1', s1)
            setattr(upsample_block, 'k2', k2)
            setattr(upsample_block, 'b2', b2)
            setattr(upsample_block, 't2', t2)
            setattr(upsample_block, 's2', s2)
            setattr(upsample_block, 'k1_1', k1_1)
            setattr(upsample_block, 'b1_1', b1_1)
            setattr(upsample_block, 't1_1', t1_1)
            setattr(upsample_block, 's1_1', s1_1)
            setattr(upsample_block, 'k2_1', k2_1)
            setattr(upsample_block, 'b2_1', b2_1)
            setattr(upsample_block, 't2_1', t2_1)
            setattr(upsample_block, 's2_1', s2_1)
            setattr(upsample_block, 'g1', g1)
            setattr(upsample_block, 'g2', g2)
            setattr(upsample_block, 'g1_1', g1_1)
            setattr(upsample_block, 'g2_1', g2_1)
            setattr(upsample_block, 'blend1', blend1)
            setattr(upsample_block, 'blend2', blend2)
            setattr(upsample_block, 'blend1_1', blend1_1)
            setattr(upsample_block, 'blend2_1', blend2_1)
            setattr(upsample_block, 'a1', a1)
            setattr(upsample_block, 'a2', a2)
            setattr(upsample_block, 'a1_1', a1_1)
            setattr(upsample_block, 'a2_1', a2_1)
            setattr(upsample_block, 'p1', p1)
            setattr(upsample_block, 'p2', p2)
            setattr(upsample_block, 'p1_1', p1_1)
            setattr(upsample_block, 'p2_1', p2_1)
            setattr(upsample_block, 'skips', skips)
            setattr(upsample_block, 'tunes', tunes)


def register_tune_crossattn_upblock2d_sdxl(
    model,
    types: int = 0,
    # 1280-channel block parameters (up_blocks[1] in SDXL)
    k1: float = 0.5,      # Backbone channel fraction for 1280ch
    b1: float = 1.2,      # Backbone LF scale - AFFECTS SEMANTICS
    t1: int = 1,          # Frequency threshold
    s1: float = 0.9,      # Backbone HF scale - AFFECTS DETAILS
    # Skip parameters for 1280ch
    k1_1: float = 0.5,    # Skip channel fraction
    b1_1: float = 1.2,    # Skip LF scale - AFFECTS COLOR/TEXTURE
    t1_1: int = 1,
    s1_1: float = 0.9,    # Skip HF scale
    # 640-channel block parameters (up_blocks[2] in SDXL)
    k2: float = 0.5,      # Controls mid-level features
    b2: float = 1.4,      # Higher value = stronger color enhancement
    t2: int = 1,
    s2: float = 0.2,      # Lower value = less high-freq noise
    k2_1: float = 0.5,
    b2_1: float = 1.4,
    t2_1: int = 1,
    s2_1: float = 0.2,
    # Global scaling
    g1: float = 1.0,
    g2: float = 1.0,
    g1_1: float = 1.0,
    g2_1: float = 1.0,
    # Blend types
    blend1: int = 0,
    blend2: int = 0,
    blend1_1: int = 0,
    blend2_1: int = 0,
    # Amplitude
    a1: float = 1.0,
    a2: float = 1.0,
    a1_1: float = 1.0,
    a2_1: float = 1.0,
    # Phase
    p1: float = 1.0,
    p2: float = 1.0,
    p1_1: float = 1.0,
    p2_1: float = 1.0,
    # Iteration control
    skips: int = 0,
    tunes: int = 0
):
    """
    Register DMFFT modifications on CrossAttnUpBlock2D blocks in SDXL U-Net.

    THIS IS THE MOST IMPORTANT FUNCTION FOR DMFFT!

    SDXL Architecture Note:
    -----------------------
    In SDXL, the CrossAttnUpBlock2D blocks are:
    - up_blocks[1]: 1280 channels - Primary semantic control
    - up_blocks[2]: 640 channels - Color/texture control

    These blocks contain BOTH:
    1. ResNet blocks for feature transformation
    2. Transformer2DModel blocks for cross-attention with text embeddings

    Key Insight from Paper:
    "CrossAttnUpBlock is the key block to modify"
    - Backbone features control SEMANTICS (what objects look like)
    - Skip features control COLOR and TEXTURE

    Recommended Settings:
    --------------------
    For enhanced image quality:
        types=4, blend=2
        b1=1.3, b2=1.3    (boost backbone LF = better semantics)
        b1_1=0.4, b2_1=0.4 (reduce skip LF = cleaner results)
        s1=1.3, s2=1.3    (boost backbone HF = sharper details)

    Args:
        model: SDXL pipeline with .unet attribute
        types: Processing mode (4 = recommended)
        See module docstring for complete parameter descriptions
    """
    def up_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            """
            Modified forward pass for CrossAttnUpBlock2D with DMFFT.

            This processes features through alternating:
            1. ResNet block (convolutional transformation)
            2. Attention block (cross-attention with text embeddings)

            DMFFT modifications are applied BEFORE the ResNet block,
            allowing frequency-domain manipulation of features that
            will then be refined by attention.
            """
            skipping = 0
            finetune = 0

            # Process each ResNet+Attention pair
            for resnet_idx, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # Capture features for debugging
                block_name = f"CrossAttnUpBlock2D_{hidden_states.shape[1]}ch"
                _capture_features(block_name, hidden_states, res_hidden_states, resnet_idx)

                # Type 1: Simple channel scaling + LF filter
                if self.types == 1:
                    if hidden_states.shape[1] == 1280:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] *= self.b1
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t1, scale=self.s1)
                    if hidden_states.shape[1] == 640:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] *= self.b2
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t2, scale=self.s2)

                # Type 2: Direct channel scaling
                elif self.types == 2:
                    if skipping >= self.skips and finetune < self.tunes:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] *= self.b2
                        res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] *= self.s2
                        finetune += 1

                # Type 3: Adaptive scaling (slow, experimental)
                elif self.types == 3:
                    hidden_states_numpy = hidden_states.to('cpu').detach().numpy()
                    hidden_states_avg = numpy.mean(hidden_states_numpy, axis=1)
                    hidden_states_max = numpy.max(hidden_states_avg, axis=(1, 2))
                    hidden_states_min = numpy.min(hidden_states_avg, axis=(1, 2))

                    if hidden_states.shape[1] == 1280:
                        hidden_states_relation = numpy.zeros((
                            hidden_states.shape[0],
                            int(hidden_states.shape[1] * self.k1),
                            hidden_states.shape[2],
                            hidden_states.shape[3]
                        ))
                        for index in range(hidden_states.shape[0]):
                            hidden_states_relation[index, :, ...] = (
                                (hidden_states_avg[index, ...] - hidden_states_min[index]) /
                                (hidden_states_max[index] - hidden_states_min[index] + 1e-8)
                            )
                        hidden_states_alpha = 1.0 + (self.b1 - 1.0) * hidden_states_relation
                        hidden_states_alpha = torch.from_numpy(hidden_states_alpha).to(hidden_states.device)
                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] *= hidden_states_alpha
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t1, scale=self.s1)

                    if hidden_states.shape[1] == 640:
                        hidden_states_relation = numpy.zeros((
                            hidden_states.shape[0],
                            int(hidden_states.shape[1] * self.k2),
                            hidden_states.shape[2],
                            hidden_states.shape[3]
                        ))
                        for index in range(hidden_states.shape[0]):
                            hidden_states_relation[index, :, ...] = (
                                (hidden_states_avg[index, ...] - hidden_states_min[index]) /
                                (hidden_states_max[index] - hidden_states_min[index] + 1e-8)
                            )
                        hidden_states_alpha = 1.0 + (self.b2 - 1.0) * hidden_states_relation
                        hidden_states_alpha = torch.from_numpy(hidden_states_alpha).to(hidden_states.device)
                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] *= hidden_states_alpha
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t2, scale=self.s2)

                # Type 4: Full DMFFT (RECOMMENDED)
                elif self.types == 4:
                    if skipping >= self.skips and finetune < self.tunes:
                        # BACKBONE: Controls semantic content
                        if self.k2 > 0.0:
                            hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = fourier_solo(
                                hidden_states[:, :int(hidden_states.shape[1] * self.k2)],
                                global_scale=self.g2,
                                freq_threshold=self.t2,
                                lf_scale=self.b2,      # Boost LF = enhance semantics
                                hf_scale=self.s2,      # Boost HF = sharper details
                                amplitude_scale=self.a2,
                                phase_scale=self.p2,
                                blend_type=self.blend2
                            )
                        # SKIP CONNECTION: Controls color/texture
                        if self.k2_1 > 0.0:
                            res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] = fourier_solo(
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)],
                                global_scale=self.g2_1,
                                freq_threshold=self.t2_1,
                                lf_scale=self.b2_1,    # Reduce LF = cleaner output
                                hf_scale=self.s2_1,    # Preserve HF = keep details
                                amplitude_scale=self.a2_1,
                                phase_scale=self.p2_1,
                                blend_type=self.blend2_1
                            )
                        finetune += 1

                # Type 5: Channel-dimension-aware DMFFT
                elif self.types == 5:
                    if skipping >= self.skips and finetune < self.tunes:
                        # 1280-channel blocks (higher-level semantics)
                        if hidden_states.shape[1] == 1280:
                            if self.k1 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = fourier_solo(
                                    hidden_states[:, :int(hidden_states.shape[1] * self.k1)],
                                    global_scale=self.g1, freq_threshold=self.t1,
                                    lf_scale=self.b1, hf_scale=self.s1,
                                    amplitude_scale=self.a1, phase_scale=self.p1,
                                    blend_type=self.blend1
                                )
                            if self.k1_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] = fourier_solo(
                                    res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)],
                                    global_scale=self.g1_1, freq_threshold=self.t1_1,
                                    lf_scale=self.b1_1, hf_scale=self.s1_1,
                                    amplitude_scale=self.a1_1, phase_scale=self.p1_1,
                                    blend_type=self.blend1_1
                                )

                        # 640-channel blocks (mid-level features)
                        if hidden_states.shape[1] == 640:
                            if self.k2 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = fourier_solo(
                                    hidden_states[:, :int(hidden_states.shape[1] * self.k2)],
                                    global_scale=self.g2, freq_threshold=self.t2,
                                    lf_scale=self.b2, hf_scale=self.s2,
                                    amplitude_scale=self.a2, phase_scale=self.p2,
                                    blend_type=self.blend2
                                )
                            if self.k2_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] = fourier_solo(
                                    res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)],
                                    global_scale=self.g2_1, freq_threshold=self.t2_1,
                                    lf_scale=self.b2_1, hf_scale=self.s2_1,
                                    amplitude_scale=self.a2_1, phase_scale=self.p2_1,
                                    blend_type=self.blend2_1
                                )
                        finetune += 1

                # Concatenate backbone with skip connection
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                # Process through ResNet + Attention
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states,
                        encoder_hidden_states,
                        None,  # timestep
                        None,  # class_labels
                        cross_attention_kwargs,
                        attention_mask,
                        encoder_attention_mask,
                        **ckpt_kwargs,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )[0]

                skipping += 1

            # Final upsampling
            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states)

            return hidden_states

        return forward

    # Apply to all CrossAttnUpBlock2D blocks
    for i, upsample_block in enumerate(model.unet.up_blocks):
        if isinstance_str(upsample_block, "CrossAttnUpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)

            # Store parameters
            setattr(upsample_block, 'types', types)
            setattr(upsample_block, 'k1', k1)
            setattr(upsample_block, 'b1', b1)
            setattr(upsample_block, 't1', t1)
            setattr(upsample_block, 's1', s1)
            setattr(upsample_block, 'k2', k2)
            setattr(upsample_block, 'b2', b2)
            setattr(upsample_block, 't2', t2)
            setattr(upsample_block, 's2', s2)
            setattr(upsample_block, 'k1_1', k1_1)
            setattr(upsample_block, 'b1_1', b1_1)
            setattr(upsample_block, 't1_1', t1_1)
            setattr(upsample_block, 's1_1', s1_1)
            setattr(upsample_block, 'k2_1', k2_1)
            setattr(upsample_block, 'b2_1', b2_1)
            setattr(upsample_block, 't2_1', t2_1)
            setattr(upsample_block, 's2_1', s2_1)
            setattr(upsample_block, 'g1', g1)
            setattr(upsample_block, 'g2', g2)
            setattr(upsample_block, 'g1_1', g1_1)
            setattr(upsample_block, 'g2_1', g2_1)
            setattr(upsample_block, 'blend1', blend1)
            setattr(upsample_block, 'blend2', blend2)
            setattr(upsample_block, 'blend1_1', blend1_1)
            setattr(upsample_block, 'blend2_1', blend2_1)
            setattr(upsample_block, 'a1', a1)
            setattr(upsample_block, 'a2', a2)
            setattr(upsample_block, 'a1_1', a1_1)
            setattr(upsample_block, 'a2_1', a2_1)
            setattr(upsample_block, 'p1', p1)
            setattr(upsample_block, 'p2', p2)
            setattr(upsample_block, 'p1_1', p1_1)
            setattr(upsample_block, 'p2_1', p2_1)
            setattr(upsample_block, 'skips', skips)
            setattr(upsample_block, 'tunes', tunes)


# Convenience aliases for backward compatibility
register_tune_upblock2d = register_tune_upblock2d_sdxl
register_tune_crossattn_upblock2d = register_tune_crossattn_upblock2d_sdxl
