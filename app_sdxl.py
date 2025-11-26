"""
DMFFT (Diffusion Model Fourier Feature Transform) - SDXL Gradio Application
============================================================================

This application provides an interactive interface for exploring DMFFT parameters
with Stable Diffusion XL (SDXL). It allows real-time comparison between baseline
SDXL generation and DMFFT-enhanced generation.

SDXL-Specific Architecture Notes:
---------------------------------
SDXL has a different U-Net structure compared to SD 1.x/2.x:

SD 2.1 up_blocks:          SDXL up_blocks:
  [0] UpBlock2D (1280)       [0] UpBlock2D (1280) - No cross-attention
  [1] CrossAttnUpBlock2D     [1] CrossAttnUpBlock2D (1280) - PRIMARY TARGET
  [2] CrossAttnUpBlock2D     [2] CrossAttnUpBlock2D (640)  - SECONDARY TARGET
  [3] CrossAttnUpBlock2D

Key Differences:
- SDXL has fewer up_blocks (3 vs 4 in SD 2.1)
- SDXL doesn't have 320-channel blocks
- SDXL's CrossAttnUpBlock2D uses Transformer2DModel instead of CrossAttention

Parameter Targeting:
-------------------
- k1, b1, s1, t1, etc. (suffix "1"): Target 1280-channel blocks
- k2, b2, s2, t2, etc. (suffix "2"): Target 640-channel blocks
- Suffix "_1" (e.g., k1_1, b1_1): Target skip connections instead of backbone

Research Paper Recommendations:
------------------------------
From the DMFFT paper experiments:

1. High-frequency scaling [1.2-1.4]:
   - Improves semantic quality and color
   - Set s1=1.3, s2=1.3 for sharper details

2. Low-frequency scaling [0.0-0.8] on skip connections:
   - Improves clarity and reduces blur
   - Set b1_1=0.4, b2_1=0.4 for cleaner results

3. Backbone LF scaling [1.2-1.4]:
   - Enhances semantic content
   - Set b1=1.3, b2=1.3 for better structure

4. CrossAttnUpBlock is the KEY block:
   - This is where semantic control happens
   - UpBlock2D has less impact on final image quality
"""

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline

from utils_sdxl import register_tune_upblock2d_sdxl, register_tune_crossattn_upblock2d_sdxl
from logger import logger
from debug_sdxl import (
    enable_debug, disable_debug, get_debugger,
    create_debug_callback, visualize_tensor_stats
)
from utils_sdxl import set_debug_capture


# ============================================================================
# SDXL Model Configuration
# ============================================================================
# SDXL models available:
# - "stabilityai/stable-diffusion-xl-base-1.0"  (Official base model)
# - "stabilityai/sdxl-turbo"  (Fast distilled version)
# - Custom fine-tuned SDXL models

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

logger.info(f"Loading SDXL model: {MODEL_ID}")
pip_model = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pip_model = pip_model.to("cuda")
# pip_model.enable_model_cpu_offload()  # Reduce VRAM usage
logger.info("Model loaded successfully!")

# Cache for baseline image (avoid regenerating when only DMFFT params change)
prompt_prev = None
seed_prev = None
steps_prev = None
sd_image_prev = None

# Global debug state
DEBUG_CONFIG = {
    "enabled": False,
    "save_latents": True,
    "save_fft": True,
    "timestep_interval": 5
}


def toggle_debug(enabled, save_latents, save_fft, timestep_interval):
    """Update global debug configuration."""
    global DEBUG_CONFIG
    DEBUG_CONFIG["enabled"] = enabled
    DEBUG_CONFIG["save_latents"] = save_latents
    DEBUG_CONFIG["save_fft"] = save_fft
    DEBUG_CONFIG["timestep_interval"] = int(timestep_interval)
    status = "enabled" if enabled else "disabled"
    return f"Debug {status}. Latents: {save_latents}, FFT: {save_fft}, Interval: {timestep_interval}"


def infer(
    prompt: str,
    seed: int,
    steps: int,
    types: int,
    # ========================================================================
    # BACKBONE PARAMETERS (hidden_states)
    # These control the main feature path through the decoder
    # ========================================================================
    k1: float,   # Channel fraction for 1280ch backbone (0.0-1.0)
    b1: float,   # LF scale for 1280ch backbone [1.2-1.4 recommended]
    t1: int,     # Frequency threshold for 1280ch (1-5 pixels)
    s1: float,   # HF scale for 1280ch backbone [1.2-1.4 for sharpness]
    k2: float,   # Channel fraction for 640ch backbone
    b2: float,   # LF scale for 640ch backbone
    t2: int,     # Frequency threshold for 640ch
    s2: float,   # HF scale for 640ch backbone
    # ========================================================================
    # SKIP CONNECTION PARAMETERS (res_hidden_states)
    # These control encoder features passed to decoder via skip connections
    # ========================================================================
    k1_1: float, # Channel fraction for 1280ch skip
    b1_1: float, # LF scale for 1280ch skip [0.0-0.8 for clarity]
    t1_1: int,   # Frequency threshold for 1280ch skip
    s1_1: float, # HF scale for 1280ch skip
    k2_1: float, # Channel fraction for 640ch skip
    b2_1: float, # LF scale for 640ch skip
    t2_1: int,   # Frequency threshold for 640ch skip
    s2_1: float, # HF scale for 640ch skip
    # ========================================================================
    # GLOBAL SCALING (applied before FFT)
    # ========================================================================
    g1: float,   # Global scale 1280ch backbone
    g2: float,   # Global scale 640ch backbone
    g1_1: float, # Global scale 1280ch skip
    g2_1: float, # Global scale 640ch skip
    # ========================================================================
    # BLEND TYPE (how amplitude/phase scaling is applied)
    # -1: No FFT, 0: All freq, 1: LF only, 2: HF only (recommended)
    # ========================================================================
    blend1: int,
    blend2: int,
    blend1_1: int,
    blend2_1: int,
    # ========================================================================
    # AMPLITUDE SCALING (Fourier magnitude)
    # Controls intensity/contrast in frequency domain
    # ========================================================================
    a1: float,
    a1_1: float,
    a2: float,
    a2_1: float,
    # ========================================================================
    # PHASE SCALING (Fourier phase)
    # Controls spatial structure - modify with caution!
    # ========================================================================
    p1: float,
    p1_1: float,
    p2: float,
    p2_1: float,
    # ========================================================================
    # ITERATION CONTROL
    # ========================================================================
    skips1: int,  # Skip first N blocks in UpBlock2D
    skips2: int,  # Skip first N blocks in CrossAttnUpBlock2D
    tunes1: int,  # Tune N blocks in UpBlock2D (max 3 in SDXL)
    tunes2: int,  # Tune N blocks in CrossAttnUpBlock2D (max 3 in SDXL)
):
    """
    Generate images with and without DMFFT enhancement.

    Returns:
        Tuple of (baseline_image, dmfft_enhanced_image)
    """
    global prompt_prev, seed_prev, steps_prev, sd_image_prev

    # Setup debug if enabled via global config
    if DEBUG_CONFIG["enabled"]:
        debugger = enable_debug(
            output_dir="debug_output",
            save_latents=DEBUG_CONFIG["save_latents"],
            save_fft=DEBUG_CONFIG["save_fft"],
            timestep_interval=DEBUG_CONFIG["timestep_interval"]
        )
        debugger.reset()
        logger.info(f"Debug enabled - output to: {debugger.gen_dir}")

    pip = pip_model

    # Check if we need to regenerate baseline
    run_baseline = False
    if prompt != prompt_prev or seed != seed_prev or steps != steps_prev:
        run_baseline = True
        prompt_prev = prompt
        seed_prev = seed
        steps_prev = steps

    # Generate baseline image (no DMFFT)
    if run_baseline:
        # Reset DMFFT to neutral settings
        register_tune_upblock2d_sdxl(
            pip,
            types=0,  # Disable DMFFT
            k1=0.0, b1=1.0, t1=0, s1=1.0,
            k2=0.0, b2=1.0, t2=0, s2=1.0,
            g1=1.0, g2=1.0,
            blend1=0, blend2=0,
            a1=1.0, a2=1.0,
            p1=1.0, p2=1.0,
            skips=0, tunes=0
        )
        register_tune_crossattn_upblock2d_sdxl(
            pip,
            types=0,
            k1=0.0, b1=1.0, t1=0, s1=1.0,
            k2=0.0, b2=1.0, t2=0, s2=1.0,
            g1=1.0, g2=1.0,
            blend1=0, blend2=0,
            a1=1.0, a2=1.0,
            p1=1.0, p2=1.0,
            skips=0, tunes=0
        )

        torch.manual_seed(seed)
        logger.info(f"Generating SDXL baseline: '{prompt}' (seed={seed}, steps={steps})")

        # Create callback for debug if enabled
        callback = create_debug_callback() if DEBUG_CONFIG["enabled"] else None

        sd_image = pip(
            prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            height=1024,
            width=1024,
            callback_on_step_end=callback,
        ).images[0]
        sd_image_prev = sd_image
    else:
        sd_image = sd_image_prev

    # ========================================================================
    # DMFFT-Enhanced Generation
    # ========================================================================
    # Apply DMFFT to UpBlock2D (affects high-level features)
    # Note: In SDXL, UpBlock2D has less impact than CrossAttnUpBlock2D
    register_tune_upblock2d_sdxl(
        pip,
        types=types,
        # 1280-channel backbone
        k1=k1, b1=b1, t1=t1, s1=s1,
        # 640-channel backbone (may not be used in SDXL UpBlock2D)
        k2=k2, b2=b2, t2=t2, s2=s2,
        # 1280-channel skip
        k1_1=k1_1, b1_1=b1_1, t1_1=t1_1, s1_1=s1_1,
        # 640-channel skip
        k2_1=k2_1, b2_1=b2_1, t2_1=t2_1, s2_1=s2_1,
        # Global
        g1=g1, g2=g2, g1_1=g1_1, g2_1=g2_1,
        # Blend
        blend1=blend1, blend2=blend2, blend1_1=blend1_1, blend2_1=blend2_1,
        # Amplitude/Phase
        a1=a1, a2=a2, a1_1=a1_1, a2_1=a2_1,
        p1=p1, p2=p2, p1_1=p1_1, p2_1=p2_1,
        # Iteration
        skips=skips1, tunes=tunes1
    )

    # Apply DMFFT to CrossAttnUpBlock2D (PRIMARY TARGET - most impact)
    register_tune_crossattn_upblock2d_sdxl(
        pip,
        types=types,
        k1=k1, b1=b1, t1=t1, s1=s1,
        k2=k2, b2=b2, t2=t2, s2=s2,
        k1_1=k1_1, b1_1=b1_1, t1_1=t1_1, s1_1=s1_1,
        k2_1=k2_1, b2_1=b2_1, t2_1=t2_1, s2_1=s2_1,
        g1=g1, g2=g2, g1_1=g1_1, g2_1=g2_1,
        blend1=blend1, blend2=blend2, blend1_1=blend1_1, blend2_1=blend2_1,
        a1=a1, a2=a2, a1_1=a1_1, a2_1=a2_1,
        p1=p1, p2=p2, p1_1=p1_1, p2_1=p2_1,
        skips=skips2, tunes=tunes2
    )

    torch.manual_seed(seed)
    logger.info(f"Generating DMFFT-enhanced: types={types}, b1={b1}, s1={s1}")

    # Reset debugger for DMFFT generation (separate from baseline)
    if DEBUG_CONFIG["enabled"]:
        debugger = get_debugger()
        debugger.reset()

    callback = create_debug_callback() if DEBUG_CONFIG["enabled"] else None

    tune_image = pip(
        prompt,
        num_inference_steps=steps,
        guidance_scale=7.5,
        height=1024,
        width=1024,
        callback_on_step_end=callback,
    ).images[0]

    # Create debug summary and GIF
    if DEBUG_CONFIG["enabled"]:
        debugger = get_debugger()
        debugger.create_latent_evolution_gif()
        debug_output = debugger.summarize_generation()
        logger.info(f"Debug output saved to: {debugger.gen_dir}, timesteps: {debug_output['timesteps_captured']}")

    return [sd_image, tune_image]


# ============================================================================
# Example Prompts
# ============================================================================
examples = [
    ["A beautiful woman with long flowing hair, professional portrait, studio lighting"],
    ["A teddy bear walking in a snowy forest, magical atmosphere"],
    ["A majestic lion with a golden mane, wildlife photography"],
    ["A cozy coffee shop interior with warm lighting and wooden furniture"],
    ["A futuristic city skyline at sunset, cyberpunk aesthetic"],
]

# ============================================================================
# Gradio Interface
# ============================================================================
block = gr.Blocks(title="DMFFT for SDXL")

with block:
    gr.Markdown("""
    # DMFFT (Diffusion Model Fourier Feature Transform) - SDXL Edition

    This tool enhances SDXL image generation using Fourier-domain feature manipulation.

    **Key Parameters:**
    - `b` (backbone LF): [1.2-1.4] enhances semantics
    - `s` (backbone HF): [1.2-1.4] improves sharpness
    - `b_1` (skip LF): [0.0-0.8] improves clarity
    - `types=4`: Recommended DMFFT mode
    - `blend=2`: Apply to high frequencies only
    """)

    with gr.Group():
        with gr.Row():
            text = gr.Textbox(
                label="Enter your prompt",
                placeholder="A beautiful woman with flowing hair...",
                lines=2
            )
            btn = gr.Button("Generate", variant="primary", scale=0)

    # ========================================================================
    # BACKBONE PARAMETERS (Most Important!)
    # ========================================================================
    with gr.Accordion("Backbone Parameters (Main Features - b/s)", open=True):
        gr.Markdown("""
        **b (LF scale)**: Controls low-frequency/coarse structure. [1.2-1.4] = enhanced semantics

        **s (HF scale)**: Controls high-frequency/fine details. [1.2-1.4] = sharper output
        """)
        with gr.Row():
            b1 = gr.Slider(label='b1 (1280ch LF)', minimum=0.0, maximum=2.0, step=0.1, value=1.3,
                          info="Backbone LF scale for 1280-channel blocks")
            b2 = gr.Slider(label='b2 (640ch LF)', minimum=0.0, maximum=2.0, step=0.1, value=1.3,
                          info="Backbone LF scale for 640-channel blocks")
        with gr.Row():
            s1 = gr.Slider(label='s1 (1280ch HF)', minimum=0.0, maximum=2.0, step=0.1, value=1.3,
                          info="Backbone HF scale for 1280-channel blocks")
            s2 = gr.Slider(label='s2 (640ch HF)', minimum=0.0, maximum=2.0, step=0.1, value=1.3,
                          info="Backbone HF scale for 640-channel blocks")

    # ========================================================================
    # SKIP CONNECTION PARAMETERS
    # ========================================================================
    with gr.Accordion("Skip Connection Parameters (Color/Texture - b_1/s_1)", open=True):
        gr.Markdown("""
        **b_1 (skip LF)**: [0.0-0.8] = cleaner, less blurry results

        **s_1 (skip HF)**: Preserves fine details from encoder
        """)
        with gr.Row():
            b1_1 = gr.Slider(label='b1_1 (1280ch skip LF)', minimum=0.0, maximum=2.0, step=0.1, value=0.4,
                            info="Skip LF scale - lower = cleaner")
            b2_1 = gr.Slider(label='b2_1 (640ch skip LF)', minimum=0.0, maximum=2.0, step=0.1, value=0.4)
        with gr.Row():
            s1_1 = gr.Slider(label='s1_1 (1280ch skip HF)', minimum=0.0, maximum=2.0, step=0.1, value=1.0,
                            info="Skip HF scale - preserves details")
            s2_1 = gr.Slider(label='s2_1 (640ch skip HF)', minimum=0.0, maximum=2.0, step=0.1, value=1.0)

    # ========================================================================
    # CHANNEL FRACTION (k) AND THRESHOLD (t)
    # ========================================================================
    with gr.Accordion("Channel Selection (k) and Frequency Threshold (t)", open=False):
        gr.Markdown("""
        **k**: Fraction of channels to modify (0.0-1.0). k=1.0 = all channels

        **t**: Frequency cutoff threshold (pixels from center). t=1-5 typical
        """)
        with gr.Row():
            k1 = gr.Slider(label='k1', minimum=0.0, maximum=1.0, step=0.1, value=1.0)
            k2 = gr.Slider(label='k2', minimum=0.0, maximum=1.0, step=0.1, value=1.0)
            k1_1 = gr.Slider(label='k1_1', minimum=0.0, maximum=1.0, step=0.1, value=1.0)
            k2_1 = gr.Slider(label='k2_1', minimum=0.0, maximum=1.0, step=0.1, value=1.0)
        with gr.Row():
            t1 = gr.Slider(label='t1', minimum=0, maximum=20, step=1, value=2)
            t2 = gr.Slider(label='t2', minimum=0, maximum=20, step=1, value=2)
            t1_1 = gr.Slider(label='t1_1', minimum=0, maximum=20, step=1, value=2)
            t2_1 = gr.Slider(label='t2_1', minimum=0, maximum=20, step=1, value=2)

    # ========================================================================
    # AMPLITUDE AND PHASE
    # ========================================================================
    with gr.Accordion("Amplitude (a) and Phase (p) Scaling", open=False):
        gr.Markdown("""
        **a (amplitude)**: Controls intensity/contrast in frequency domain

        **p (phase)**: Controls spatial structure - modify carefully!
        """)
        with gr.Row():
            a1 = gr.Slider(label='a1', minimum=0.0, maximum=2.0, step=0.1, value=1.0)
            a1_1 = gr.Slider(label='a1_1', minimum=0.0, maximum=2.0, step=0.1, value=1.0)
            a2 = gr.Slider(label='a2', minimum=0.0, maximum=2.0, step=0.1, value=1.0)
            a2_1 = gr.Slider(label='a2_1', minimum=0.0, maximum=2.0, step=0.1, value=1.0)
        with gr.Row():
            p1 = gr.Slider(label='p1', minimum=0.0, maximum=2.0, step=0.1, value=1.0)
            p1_1 = gr.Slider(label='p1_1', minimum=0.0, maximum=2.0, step=0.1, value=1.0)
            p2 = gr.Slider(label='p2', minimum=0.0, maximum=2.0, step=0.1, value=1.0)
            p2_1 = gr.Slider(label='p2_1', minimum=0.0, maximum=2.0, step=0.1, value=1.0)

    # ========================================================================
    # GLOBAL SCALING
    # ========================================================================
    with gr.Accordion("Global Scaling (g) - Pre-FFT Multiplier", open=False):
        with gr.Row():
            g1 = gr.Slider(label='g1', minimum=0.0, maximum=2.0, step=0.1, value=1.0)
            g2 = gr.Slider(label='g2', minimum=0.0, maximum=2.0, step=0.1, value=1.0)
            g1_1 = gr.Slider(label='g1_1', minimum=0.0, maximum=2.0, step=0.1, value=1.0)
            g2_1 = gr.Slider(label='g2_1', minimum=0.0, maximum=2.0, step=0.1, value=1.0)

    # ========================================================================
    # OUTPUT IMAGES
    # ========================================================================
    with gr.Row():
        with gr.Column():
            image_1 = gr.Image(label="SDXL Baseline (No DMFFT)")
        with gr.Column():
            image_2 = gr.Image(label="DMFFT Enhanced")

    # ========================================================================
    # CONTROL PARAMETERS
    # ========================================================================
    with gr.Accordion("Generation Settings", open=True):
        with gr.Row():
            seed = gr.Slider(label='Seed', minimum=0, maximum=1000, step=1, value=42)
            steps = gr.Slider(label='Steps', minimum=10, maximum=100, step=5, value=30)

        with gr.Row():
            types = gr.Slider(
                label='DMFFT Mode (types)',
                minimum=0, maximum=5, step=1, value=4,
                info="0=off, 4=recommended (full FFT), 5=channel-aware"
            )

        gr.Markdown("**Blend Type:** -1=no FFT, 0=all freq, 1=LF only, **2=HF only (recommended)**, 3/4=mixed")
        with gr.Row():
            blend1 = gr.Slider(label='blend1', minimum=-1, maximum=4, step=1, value=2)
            blend2 = gr.Slider(label='blend2', minimum=-1, maximum=4, step=1, value=2)
            blend1_1 = gr.Slider(label='blend1_1', minimum=-1, maximum=4, step=1, value=2)
            blend2_1 = gr.Slider(label='blend2_1', minimum=-1, maximum=4, step=1, value=2)

    with gr.Accordion("Iteration Control (skips/tunes)", open=False):
        gr.Markdown("""
        **skips**: Number of ResNet blocks to skip before applying DMFFT

        **tunes**: Number of ResNet blocks to apply DMFFT to (max 3 in SDXL)
        """)
        with gr.Row():
            skips1 = gr.Slider(label='skips1 (UpBlock2D)', minimum=0, maximum=10, step=1, value=0)
            skips2 = gr.Slider(label='skips2 (CrossAttnUpBlock2D)', minimum=0, maximum=10, step=1, value=0)
        with gr.Row():
            tunes1 = gr.Slider(label='tunes1 (UpBlock2D)', minimum=0, maximum=10, step=1, value=3)
            tunes2 = gr.Slider(label='tunes2 (CrossAttnUpBlock2D)', minimum=0, maximum=10, step=1, value=3)

    # ========================================================================
    # DEBUG VISUALIZATION
    # ========================================================================
    with gr.Accordion("Debug Visualization", open=False):
        gr.Markdown("""
        **Debug Mode** captures internal U-Net features during generation:
        - **Latent Evolution**: See how the latent changes across denoising steps
        - **FFT Analysis**: Visualize frequency domain of feature maps
        - Output saved to `debug_output/` folder with animated GIFs
        """)
        with gr.Row():
            debug_enabled = gr.Checkbox(label="Enable Debug", value=False,
                                       info="Capture latents and feature maps")
            debug_save_latents = gr.Checkbox(label="Save Latents", value=True,
                                            info="Save latent at each timestep")
            debug_save_fft = gr.Checkbox(label="Save FFT", value=True,
                                        info="Save frequency analysis")
        with gr.Row():
            debug_timestep_interval = gr.Slider(
                label='Timestep Interval',
                minimum=1, maximum=10, step=1, value=5,
                info="Save every N timesteps (lower = more detail, larger files)"
            )

        debug_output = gr.Textbox(
            label="Debug Info",
            lines=3,
            interactive=False,
            placeholder="Debug output will appear here..."
        )

    # ========================================================================
    # EXAMPLES
    # ========================================================================
    gr.Examples(
        examples=examples,
        inputs=[text],
        outputs=[image_1, image_2],
    )

    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================
    all_inputs = [
        text, seed, steps, types,
        k1, b1, t1, s1,
        k2, b2, t2, s2,
        k1_1, b1_1, t1_1, s1_1,
        k2_1, b2_1, t2_1, s2_1,
        g1, g2, g1_1, g2_1,
        blend1, blend2, blend1_1, blend2_1,
        a1, a1_1, a2, a2_1,
        p1, p1_1, p2, p2_1,
        skips1, skips2, tunes1, tunes2
    ]
    outputs = [image_1, image_2]

    text.submit(infer, inputs=all_inputs, outputs=outputs)
    btn.click(infer, inputs=all_inputs, outputs=outputs)

    # Debug toggle handler (separate from main inference)
    debug_inputs = [debug_enabled, debug_save_latents, debug_save_fft, debug_timestep_interval]
    debug_enabled.change(toggle_debug, inputs=debug_inputs, outputs=debug_output)
    debug_save_latents.change(toggle_debug, inputs=debug_inputs, outputs=debug_output)
    debug_save_fft.change(toggle_debug, inputs=debug_inputs, outputs=debug_output)
    debug_timestep_interval.change(toggle_debug, inputs=debug_inputs, outputs=debug_output)

# ============================================================================
# Launch
# ============================================================================
if __name__ == "__main__":
    block.launch(server_name="0.0.0.0", server_port=7860)
