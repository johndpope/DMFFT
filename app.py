"""
DMFFT (Diffusion Model Fourier Feature Transform) - Gradio Application
=======================================================================

This application provides an interactive interface for exploring DMFFT parameters
with Stable Diffusion. It allows real-time comparison between baseline generation
and DMFFT-enhanced generation.

================================================================================
DMFFT PARAMETER REFERENCE GUIDE
================================================================================

U-NET ARCHITECTURE (SD 2.1):
----------------------------
The U-Net decoder has 4 up_blocks that progressively upsample features:

  up_blocks[0]: UpBlock2D         - 1280 channels (highest level)
  up_blocks[1]: CrossAttnUpBlock2D - 1280 channels (with cross-attention)
  up_blocks[2]: CrossAttnUpBlock2D - 640 channels
  up_blocks[3]: CrossAttnUpBlock2D - 320 channels (lowest level)

KEY INSIGHT: CrossAttnUpBlock is the PRIMARY target for quality enhancement!
- Backbone features (hidden_states) → Control SEMANTICS (object structure)
- Skip features (res_hidden_states) → Control COLOR and TEXTURE

================================================================================
PARAMETER NAMING CONVENTION
================================================================================

Each parameter has a suffix that indicates what it targets:

  Suffix "1":   Targets 1280-channel blocks (high-level semantics)
  Suffix "2":   Targets 640-channel blocks (mid-level features)
  Suffix "_1":  Targets skip connections (encoder → decoder pathway)
  No "_1":      Targets backbone (main decoder pathway)

Examples:
  b1    = Backbone LF scale for 1280-channel blocks
  b1_1  = Skip connection LF scale for 1280-channel blocks
  b2    = Backbone LF scale for 640-channel blocks
  b2_1  = Skip connection LF scale for 640-channel blocks

================================================================================
CHANNEL SELECTION (k parameters) - Which channels to modify
================================================================================

  k1:    Fraction of backbone channels to modify in 1280ch blocks (0.0-1.0)
  k2:    Fraction of backbone channels to modify in 640ch blocks
  k1_1:  Fraction of skip channels to modify in 1280ch blocks
  k2_1:  Fraction of skip channels to modify in 640ch blocks

  - k=0.0: No channels modified (DMFFT disabled for this path)
  - k=0.5: First 50% of channels modified
  - k=1.0: All channels modified (full DMFFT effect)

  Higher k = More global changes to structure/style

================================================================================
BACKBONE SCALING (b/s parameters) - Control main decoder features
================================================================================

LOW-FREQUENCY SCALING (b = "backbone LF"):
  b1:    LF scale for 1280ch backbone → Controls coarse structure
  b2:    LF scale for 640ch backbone → Controls shape/form

  RECOMMENDED VALUES: [1.2 - 1.4]
  - b > 1.0: Amplifies low-frequency (enhances semantic structure)
  - b < 1.0: Reduces coarse structure (softer output)
  - Paper finding: b=1.3 gives good semantic enhancement

HIGH-FREQUENCY SCALING (s = "backbone HF"):
  s1:    HF scale for 1280ch backbone → Controls fine semantic details
  s2:    HF scale for 640ch backbone → Controls texture/edges

  RECOMMENDED VALUES: [1.2 - 1.4]
  - s > 1.0: Sharpens high-frequency details
  - s < 1.0: Smooths/blurs details
  - Paper finding: s=1.3 improves sharpness

================================================================================
SKIP CONNECTION SCALING (b_1/s_1 parameters) - Control encoder features
================================================================================

SKIP LF SCALING (b_1):
  b1_1:  Skip LF scale for 1280ch → Color consistency at high level
  b2_1:  Skip LF scale for 640ch → Color/texture distribution

  RECOMMENDED VALUES: [0.0 - 0.8] for CLARITY
  - Lower values = Cleaner, less blurry results
  - Paper finding: b_1=0.4 significantly improves clarity

SKIP HF SCALING (s_1):
  s1_1:  Skip HF scale for 1280ch → Fine detail preservation
  s2_1:  Skip HF scale for 640ch → Texture detail from encoder

  RECOMMENDED VALUES: [0.8 - 1.2]
  - s_1 ≈ 1.0: Preserve original encoder details
  - s_1 > 1.0: Enhance encoder texture details

================================================================================
FREQUENCY THRESHOLD (t parameters) - Define low/high frequency boundary
================================================================================

  t1:    Threshold for 1280ch backbone (pixels from spectrum center)
  t2:    Threshold for 640ch backbone
  t1_1:  Threshold for 1280ch skip
  t2_1:  Threshold for 640ch skip

  - t=1: Very small LF region (mostly HF manipulation)
  - t=5: Moderate LF region
  - t=10: Large LF region

  RECOMMENDED VALUES: 1-5 for most applications

================================================================================
GLOBAL SCALING (g parameters) - Pre-FFT multiplier
================================================================================

  g1:    Global scale for 1280ch backbone (applied BEFORE FFT)
  g2:    Global scale for 640ch backbone
  g1_1:  Global scale for 1280ch skip
  g2_1:  Global scale for 640ch skip

  - g=1.0: No global change (default)
  - g > 1.0: Overall feature amplification
  - g < 1.0: Overall feature reduction

================================================================================
AMPLITUDE SCALING (a parameters) - Fourier magnitude control
================================================================================

  a1:    Amplitude scale for 1280ch backbone
  a2:    Amplitude scale for 640ch backbone
  a1_1:  Amplitude scale for 1280ch skip
  a2_1:  Amplitude scale for 640ch skip

  - Amplitude = magnitude of frequency components
  - a > 1.0: Increases contrast/intensity
  - a < 1.0: Reduces contrast

================================================================================
PHASE SCALING (p parameters) - Fourier phase control (USE WITH CAUTION!)
================================================================================

  p1:    Phase scale for 1280ch backbone
  p2:    Phase scale for 640ch backbone
  p1_1:  Phase scale for 1280ch skip
  p2_1:  Phase scale for 640ch skip

  - Phase encodes SPATIAL POSITION and STRUCTURE
  - p=1.0: Preserve original phase (RECOMMENDED)
  - p≠1.0: May cause spatial distortions

================================================================================
BLEND TYPE - How amplitude/phase scaling is applied
================================================================================

  blend1, blend2, blend1_1, blend2_1

  Values:
    -1: No FFT applied (only global_scale used)
     0: Apply a/p scaling to ENTIRE frequency spectrum
     1: Apply a/p scaling only to LOW frequencies
     2: Apply a/p scaling only to HIGH frequencies (RECOMMENDED)
     3: LF amplitude + HF phase scaling
     4: HF amplitude + LF phase scaling

================================================================================
ITERATION CONTROL - Which ResNet blocks to modify
================================================================================

  skips:   Number of ResNet blocks to SKIP before applying DMFFT
  tunes:   Number of ResNet blocks to APPLY DMFFT to (after skips)

  Example with skips=1, tunes=2:
    Block 0: SKIP (no DMFFT)
    Block 1: TUNE (DMFFT applied)
    Block 2: TUNE (DMFFT applied)
    Block 3+: No more tuning

================================================================================
TYPE MODES - Processing algorithm selection
================================================================================

  types=0: No modification (baseline)
  types=1: Simple channel scaling + fourier_filter on skip
  types=2: Direct channel scaling without FFT
  types=3: Adaptive scaling based on feature statistics (slow)
  types=4: Full fourier_solo (RECOMMENDED for DMFFT)
  types=5: Channel-dimension-aware fourier_solo

================================================================================
RECOMMENDED SETTINGS FOR IMAGE ENHANCEMENT
================================================================================

For general quality improvement:
  types=4, blend=2 (apply HF scaling only)
  k1=0.5, k2=0.5, k1_1=1.0, k2_1=1.0
  b1=1.3, b2=1.3       (enhance backbone LF for better semantics)
  b1_1=0.4, b2_1=0.4   (reduce skip LF for clarity)
  s1=1.3, s2=1.3       (enhance backbone HF for sharpness)
  s1_1=1.0, s2_1=1.0   (preserve skip HF details)
  t1=2, t2=2
  a1=1.0, a2=1.0, p1=1.0, p2=1.0  (neutral amplitude/phase)
  g1=1.0, g2=1.0
  skips=0, tunes=3

For maximum sharpness:
  Increase s1, s2 to 1.4-1.5

For better color consistency:
  Increase b1, b2 to 1.4
  Decrease b1_1, b2_1 to 0.2-0.3

================================================================================
"""

import gradio as gr

import torch

from diffusers import StableDiffusionPipeline

from utils import register_tune_upblock2d, register_tune_crossattn_upblock2d


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Available Stable Diffusion models:
# model_id = "stabilityai/stable-diffusion-v1-1"  # Original SD 1.1
# model_id = "stabilityai/stable-diffusion-v1-2"  # SD 1.2
# model_id = "stabilityai/stable-diffusion-v1-3"  # SD 1.3
# model_id = "stabilityai/stable-diffusion-v1-4"  # SD 1.4 (popular)
# model_id = "stabilityai/stable-diffusion-v1-5"  # SD 1.5 (most used)
model_id = "stabilityai/stable-diffusion-2-1"     # SD 2.1 (higher resolution)

pip_model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pip_model = pip_model.to("cuda")

prompt_prev = None
seed_prev = None
steps_prev = None
sd_image_prev = None


def infer(prompt, seed, steps,
          types,
          k1, b1, t1, s1,
          k2, b2, t2, s2,
          k1_1, b1_1, t1_1, s1_1,
          k2_1, b2_1, t2_1, s2_1,
          g1, g2, g1_1, g2_1,
          blend1, blend2, blend1_1, blend2_1,
          a1, a1_1, a2, a2_1,
          p1, p1_1, p2, p2_1,
          skips1, skips2, tunes1, tunes2):
    global prompt_prev
    global seed_prev
    global steps_prev
    global sd_image_prev

    pip = pip_model

    run_baseline = False
    if prompt != prompt_prev or seed != seed_prev or steps != steps_prev:
        run_baseline = True
        prompt_prev = prompt
        seed_prev = seed
        steps_prev = steps

    if run_baseline:
        for seed in range(1000):
            if seed == 880:
                register_tune_upblock2d(pip,
                                        types=0,
                                        k1=0.0, b1=1.0, t1=0, s1=1.0,
                                        k2=0.0, b2=1.0, t2=0, s2=1.0,
                                        g1=1.0, g2=1.0,
                                        blend1=0, blend2=0,
                                        a1=1.0, a2=1.0,
                                        p1=1.0, p2=1.0,
                                        skips=0, tunes=0)
                register_tune_crossattn_upblock2d(pip,
                                                  types=0,
                                                  k1=0.0, b1=1.0, t1=0, s1=1.0,
                                                  k2=0.0, b2=1.0, t2=0, s2=1.0,
                                                  g1=1.0, g2=1.0,
                                                  blend1=0, blend2=0,
                                                  a1=1.0, a2=1.0,
                                                  p1=1.0, p2=1.0,
                                                  skips=0, tunes=0)

                torch.manual_seed(seed)
                print("Generating SD:")

                sd_image = pip(prompt, num_inference_steps=steps).images[0]
                sd_image_prev = sd_image
    else:
        sd_image = sd_image_prev

    for groups in range(1):
        # k1 = 0.5
        # k1_1 = 1.0
        # k2 = 0.5
        # k2_1 = 1.0

        # k1 = 0.5
        # k1_1 = 0.5
        # k2 = 0.5
        # k2_1 = 0.5

        # k1 = 1.0
        # k1_1 = 1.0
        # k2 = 1.0
        # k2_1 = 1.0

        seed = 880
        types = 4  # 4 1

        blend1 = 2
        blend1_1 = 2
        blend2 = 2
        blend2_1 = 2

        # s1 = groups / 10.0
        # s1_1 = groups / 10.0
        # s2 = groups / 10.0
        # s2_1 = groups / 10.0

        # g1 = groups / 10.0
        # g1_1 = groups / 10.0
        # g2 = groups / 10.0
        # g2_1 = groups / 10.0

        # s1_1 = 1.5
        # s2_1 = 1.5

        # g1 = 0.5
        # g1_1 = 1.0
        # g2 = 1.5
        # g2_1 = 1.0

        # b1 = 1.0
        # b1_1 = 1.5
        # b2 = 1.4
        # b2_1 = 1.5

        # b1 = groups / 10.0
        # b1_1 = groups / 10.0
        # b2 = groups / 10.0
        # b2_1 = groups / 10.0

        # a1 = groups / 10.0
        # a1_1 = groups / 10.0
        # a2 = groups / 10.0
        # a2_1 = groups / 10.0

        # p1 = groups / 10.0
        # p1_1 = groups / 10.0
        # p2 = groups / 10.0
        # p2_1 = groups / 10.0

        for aid in range(8):  # 20 8 21 22 23 24
            for analysis in range(20):  # 20
                a1 = 1.0
                a1_1 = 1.0

                if aid != 22:
                    a2 = 1.0

                a2_1 = 1.0

                p1 = 1.0
                p1_1 = 1.0
                p2 = 1.0
                p2_1 = 1.0

                if aid != 20:
                    b1 = 1.0

                b1_1 = 1.0
                b2 = 1.0
                b2_1 = 1.0

                if aid != 21:
                    s1 = 1.0

                s1_1 = 1.0
                s2 = 1.0
                s2_1 = 1.0

                g1 = 1.0
                g1_1 = 1.0
                g2 = 1.0
                g2_1 = 1.0

                if aid == 0:
                    a1 = analysis / 10.0
                elif aid == 1:
                    a1_1 = analysis / 10.0
                elif aid == 2:
                    a2 = analysis / 10.0
                elif aid == 3:
                    a2_1 = analysis / 10.0
                elif aid == 4:
                    p1 = analysis / 10.0
                elif aid == 5:
                    p1_1 = analysis / 10.0
                elif aid == 6:
                    p2 = analysis / 10.0
                elif aid == 7:
                    p2_1 = analysis / 10.0
                elif aid == 8:
                    b1 = analysis / 10.0
                elif aid == 9:
                    b1_1 = analysis / 10.0
                elif aid == 10:
                    b2 = analysis / 10.0
                elif aid == 11:
                    b2_1 = analysis / 10.0
                elif aid == 12:
                    s1 = analysis / 10.0
                elif aid == 13:
                    s1_1 = analysis / 10.0
                elif aid == 14:
                    s2 = analysis / 10.0
                elif aid == 15:
                    s2_1 = analysis / 10.0
                elif aid == 16:
                    g1 = analysis / 10.0
                elif aid == 17:
                    g1_1 = analysis / 10.0
                elif aid == 18:
                    g2 = analysis / 10.0
                elif aid == 19:
                    g2_1 = analysis / 10.0
                elif aid == 20:
                    b2 = analysis / 10.0
                elif aid == 21:
                    s2 = analysis / 10.0
                elif aid == 22:
                    a2_1 = analysis / 10.0

                # s1 = analysis / 10.0
                # s1_1 = analysis / 10.0
                # s2 = analysis / 10.0
                # s2_1 = analysis / 10.0

                # s1 = 1.4
                # s1_1 = 1.4
                # s2 = 1.4
                # s2_1 = 1.4

                # p1 = analysis / 10.0
                # p1_1 = analysis / 10.0
                # p2 = analysis / 10.0
                # p2_1 = analysis / 10.0

                # b1 = analysis / 10.0
                # b1_1 = analysis / 10.0
                # b2 = analysis / 10.0
                # b2_1 = analysis / 10.0

                # a1 = analysis / 10.0
                # a1_1 = analysis / 10.0
                # a2 = analysis / 10.0
                # a2_1 = analysis / 10.0

                # g1 = analysis / 10.0
                # g1_1 = analysis / 10.0
                # g2 = analysis / 10.0
                # g2_1 = analysis / 10.0

                # b1 = 2.8
                # s1 = 0.1

                # tunes1 = analysis
                # skips1 = analysis

                # tunes2 = analysis
                # skips2 = analysis

                register_tune_upblock2d(pip,
                                        types=types,
                                        k1=k1, b1=b1, t1=t1, s1=s1,
                                        k1_1=k1_1, b1_1=b1_1, t1_1=t1_1, s1_1=s1_1,
                                        k2=k2, b2=b2, t2=t2, s2=s2,
                                        k2_1=k2_1, b2_1=b2_1, t2_1=t2_1, s2_1=s2_1,
                                        g1=g1, g2=g2, g1_1=g1_1, g2_1=g2_1,
                                        blend1=blend1, blend1_1=blend1_1, blend2=blend2, blend2_1=blend2_1,
                                        a1=a1, a1_1=a1_1, a2=a2, a2_1=a2_1,
                                        p1=p1, p1_1=p1_1, p2=p2, p2_1=p2_1,
                                        skips=skips1, tunes=tunes1)

                register_tune_crossattn_upblock2d(pip,
                                                  types=types,
                                                  k1=k1, b1=b1, t1=t1, s1=s1,
                                                  k1_1=k1_1, b1_1=b1_1, t1_1=t1_1, s1_1=s1_1,
                                                  k2=k2, b2=b2, t2=t2, s2=s2,
                                                  k2_1=k2_1, b2_1=b2_1, t2_1=t2_1, s2_1=s2_1,
                                                  g1=g1, g2=g2, g1_1=g1_1, g2_1=g2_1,
                                                  blend1=blend1, blend1_1=blend1_1, blend2=blend2, blend2_1=blend2_1,
                                                  a1=a1, a1_1=a1_1, a2=a2, a2_1=a2_1,
                                                  p1=p1, p1_1=p1_1, p2=p2, p2_1=p2_1,
                                                  skips=skips2, tunes=tunes2)

                torch.manual_seed(seed)
                print("Generating:")

                tune_image = pip(prompt, num_inference_steps=steps).images[0]

                if aid == 0:
                    tune_image.save('./a1/' + str(a1) + ".png")
                elif aid == 1:
                    tune_image.save('./a1_1/' + str(a1_1) + ".png")
                elif aid == 2:
                    tune_image.save('./a2/' + str(a2) + ".png")
                elif aid == 3:
                    tune_image.save('./a2_1/' + str(a2_1) + ".png")
                elif aid == 4:
                    tune_image.save('./p1/' + str(p1) + ".png")
                elif aid == 5:
                    tune_image.save('./p1_1/' + str(p1_1) + ".png")
                elif aid == 6:
                    tune_image.save('./p2/' + str(p2) + ".png")
                elif aid == 7:
                    tune_image.save('./p2_1/' + str(p2_1) + ".png")
                elif aid == 8:
                    tune_image.save('./b1/' + str(b1) + ".png")
                elif aid == 9:
                    tune_image.save('./b1_1/' + str(b1_1) + ".png")
                elif aid == 10:
                    tune_image.save('./b2/' + str(b2) + ".png")
                elif aid == 11:
                    tune_image.save('./b2_1/' + str(b2_1) + ".png")
                elif aid == 12:
                    tune_image.save('./s1/' + str(s1) + ".png")
                elif aid == 13:
                    tune_image.save('./s1_1/' + str(s1_1) + ".png")
                elif aid == 14:
                    tune_image.save('./s2/' + str(s2) + ".png")
                elif aid == 15:
                    tune_image.save('./s2_1/' + str(s2_1) + ".png")
                elif aid == 16:
                    tune_image.save('./g1/' + str(g1) + ".png")
                elif aid == 17:
                    tune_image.save('./g1_1/' + str(g1_1) + ".png")
                elif aid == 18:
                    tune_image.save('./g2/' + str(g2) + ".png")
                elif aid == 19:
                    tune_image.save('./g2_1/' + str(g2_1) + ".png")
                elif aid == 20:
                    tune_image.save('./b1 & b2/' + str(b1) + "-" + str(b2) + ".png")
                elif aid == 21:
                    tune_image.save('./s1 & s2/' + str(s1) + "-" + str(s2) + ".png")
                elif aid == 22:
                    tune_image.save('./a2 & a2_1/' + str(a2) + "-" + str(a2_1) + ".png")

    images = [sd_image, tune_image]

    return images


examples = [
    [
        "Girl with black hair and blue eyes.",
    ],
    [
        "A teddy bear walking in the snowstorm.",
    ],
    [
        "A cat riding a motorcycle.",
    ],
]

block = gr.Blocks(css='style.css')

with block:
    with gr.Group():
        with gr.Row(elem_id="prompt-container"):
            with gr.Column():
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                    )
            btn = gr.Button("Generate image", scale=0)

    with gr.Group():
        with gr.Row():
            with gr.Accordion('Parameters: ', open=True):
                with gr.Row():
                    b1 = gr.Slider(label='b1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                    b1_1 = gr.Slider(label='b1_1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)

                with gr.Row():
                    b2 = gr.Slider(label='b2', minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                    b2_1 = gr.Slider(label='b2_1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)

                with gr.Row():
                    s1 = gr.Slider(label='s1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                    s1_1 = gr.Slider(label='s1_1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)

                with gr.Row():
                    s2 = gr.Slider(label='s2', minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                    s2_1 = gr.Slider(label='s2_1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)

    with gr.Group():
        with gr.Row():
            with gr.Accordion('Parameters: ', open=True):
                with gr.Row():
                    t1 = gr.Slider(label='t1', minimum=0, maximum=20, step=1, value=1)
                    t1_1 = gr.Slider(label='t1_1', minimum=0, maximum=20, step=1, value=1)

                with gr.Row():
                    t2 = gr.Slider(label='t2', minimum=0, maximum=20, step=1, value=1)
                    t2_1 = gr.Slider(label='t2_1', minimum=0, maximum=20, step=1, value=1)

                with gr.Row():
                    k1 = gr.Slider(label='k1', minimum=0.0, maximum=1.0, step=0.1, value=1.0)
                    k1_1 = gr.Slider(label='k1_1', minimum=0.0, maximum=1.0, step=0.1, value=1.0)

                with gr.Row():
                    k2 = gr.Slider(label='k2', minimum=0.0, maximum=1.0, step=0.1, value=1.0)
                    k2_1 = gr.Slider(label='k2_1', minimum=0.0, maximum=1.0, step=0.1, value=1.0)

    with gr.Group():
        with gr.Row():
            with gr.Accordion('Parameters: ', open=True):
                with gr.Row():
                    g1 = gr.Slider(label='g1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                    g1_1 = gr.Slider(label='g1_1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)

                with gr.Row():
                    g2 = gr.Slider(label='g2', minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                    g2_1 = gr.Slider(label='g2_1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)

    with gr.Group():
        with gr.Row():
            with gr.Accordion('Parameters: ', open=True):
                with gr.Row():
                    a1 = gr.Slider(label='a1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                    a1_1 = gr.Slider(label='a1_1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)

                with gr.Row():
                    a2 = gr.Slider(label='a2', minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                    a2_1 = gr.Slider(label='a2_1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)

                with gr.Row():
                    p1 = gr.Slider(label='p1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                    p1_1 = gr.Slider(label='p1_1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)

                with gr.Row():
                    p2 = gr.Slider(label='p2', minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                    p2_1 = gr.Slider(label='p2_1', minimum=0.0, maximum=3.0, step=0.1, value=1.0)

    with gr.Row():
        with gr.Group():
            with gr.Row():
                with gr.Column() as I1:
                    image_1 = gr.Image(interactive=False)
                    image_1_label = gr.Markdown("SD")

        with gr.Group():
            with gr.Row():
                with gr.Column() as I2:
                    image_2 = gr.Image(interactive=False)
                    image_2_label = gr.Markdown("Tune")

    with gr.Group():
        with gr.Row():
            with gr.Accordion('Parameters: ', open=True):
                with gr.Row():
                    seed = gr.Slider(label='seed', minimum=0, maximum=1000, step=1, value=0)
                    steps = gr.Slider(label='steps', minimum=1, maximum=200, step=1, value=25)

                with gr.Row():
                    types = gr.Slider(label='types', minimum=0, maximum=5, step=1, value=0)

                with gr.Row():
                    blend1 = gr.Slider(label='blend1', minimum=0, maximum=5, step=1, value=0)
                    blend1_1 = gr.Slider(label='blend1_1', minimum=0, maximum=5, step=1, value=0)

                with gr.Row():
                    blend2 = gr.Slider(label='blend2', minimum=0, maximum=5, step=1, value=0)
                    blend2_1 = gr.Slider(label='blend2_1', minimum=0, maximum=5, step=1, value=0)

    with gr.Group():
        with gr.Row():
            with gr.Accordion('Parameters: ', open=True):
                with gr.Row():
                    skips1 = gr.Slider(label='skips1', minimum=0, maximum=20, step=1, value=0)
                    skips2 = gr.Slider(label='skips2', minimum=0, maximum=20, step=1, value=0)

                with gr.Row():
                    tunes1 = gr.Slider(label='tunes1', minimum=0, maximum=20, step=1, value=10)
                    tunes2 = gr.Slider(label='tunes2', minimum=0, maximum=20, step=1, value=10)

    ex = gr.Examples(examples=examples, fn=infer, inputs=[text, seed, steps, types, k1, b1, t1, s1, k2, b2, t2, s2, k1_1, b1_1, t1_1, s1_1, k2_1, b2_1, t2_1, s2_1, g1, g2, g1_1, g2_1, blend1, blend2, blend1_1, blend2_1, a1, a1_1, a2, a2_1, p1, p1_1, p2, p2_1, skips1, skips2, tunes1, tunes2], outputs=[image_1, image_2], cache_examples=False)
    ex.dataset.headers = [""]

    text.submit(infer, inputs=[text, seed, steps, types, k1, b1, t1, s1, k2, b2, t2, s2, k1_1, b1_1, t1_1, s1_1, k2_1, b2_1, t2_1, s2_1, g1, g2, g1_1, g2_1, blend1, blend2, blend1_1, blend2_1, a1, a1_1, a2, a2_1, p1, p1_1, p2, p2_1, skips1, skips2, tunes1, tunes2], outputs=[image_1, image_2])
    btn.click(infer, inputs=[text, seed, steps, types, k1, b1, t1, s1, k2, b2, t2, s2, k1_1, b1_1, t1_1, s1_1, k2_1, b2_1, t2_1, s2_1, g1, g2, g1_1, g2_1, blend1, blend2, blend1_1, blend2_1, a1, a1_1, a2, a2_1, p1, p1_1, p2, p2_1, skips1, skips2, tunes1, tunes2], outputs=[image_1, image_2])

block.launch()
