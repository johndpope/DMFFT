#!/usr/bin/env python3
"""
Comprehensive exploration of SDXL UNet architecture
Maps out all attention blocks and tracks latent resolutions
"""

import torch
from diffusers import StableDiffusionXLPipeline

def explore_block_attention_layers(block, block_name, attention_blocks):
    """Recursively explore attention layers in a block."""
    if hasattr(block, 'attentions'):
        for att_idx, attention in enumerate(block.attentions):
            if hasattr(attention, 'transformer_blocks'):
                for tb_idx, tb in enumerate(attention.transformer_blocks):
                    if hasattr(tb, 'attn1'):
                        key = f"{block_name}.attentions.{att_idx}.transformer_blocks.{tb_idx}.attn1"
                        attention_blocks[key] = {
                            'has_to_q': hasattr(tb.attn1, 'to_q'),
                            'has_to_k': hasattr(tb.attn1, 'to_k'),
                            'has_to_v': hasattr(tb.attn1, 'to_v'),
                        }
                        if hasattr(tb.attn1, 'to_q'):
                            attention_blocks[key]['q_shape'] = tb.attn1.to_q.weight.shape
                    if hasattr(tb, 'attn2'):
                        key = f"{block_name}.attentions.{att_idx}.transformer_blocks.{tb_idx}.attn2"
                        attention_blocks[key] = {
                            'has_to_q': hasattr(tb.attn2, 'to_q'),
                            'has_to_k': hasattr(tb.attn2, 'to_k'),
                            'has_to_v': hasattr(tb.attn2, 'to_v'),
                        }
                        if hasattr(tb.attn2, 'to_q'):
                            attention_blocks[key]['q_shape'] = tb.attn2.to_q.weight.shape

def trace_resolution_flow(unet):
    """Trace how resolutions change through the UNet."""
    # SDXL starts with 128x128 latent for 1024x1024 image (8x downsampling)
    print("\n" + "="*60)
    print("RESOLUTION FLOW THROUGH SDXL UNET")
    print("="*60)
    print("\nFor 1024x1024 image -> 128x128 latent input (4 channels)")

    resolutions = {}
    current_res = 128

    print("\n📥 ENCODER (Downsampling) Path:")
    print("-" * 40)

    # Down blocks
    down_block_configs = [
        (0, 128, False),  # No downsampling
        (1, 128, True),   # Downsample to 64
        (2, 64, True),    # Downsample to 32
    ]

    for block_idx, (idx, res, downsample) in enumerate(down_block_configs):
        if hasattr(unet, 'down_blocks'):
            block = unet.down_blocks[idx]
            block_name = f"down_blocks.{idx}"

            # Count attention layers
            num_attentions = len(block.attentions) if hasattr(block, 'attentions') else 0
            num_resnets = len(block.resnets) if hasattr(block, 'resnets') else 0

            print(f"\n{block_name}:")
            print(f"  Resolution: {res}x{res}")
            print(f"  ResNet layers: {num_resnets}")
            print(f"  Attention layers: {num_attentions}")

            if num_attentions > 0 and hasattr(block.attentions[0], 'transformer_blocks'):
                num_transformer_blocks = len(block.attentions[0].transformer_blocks)
                print(f"  Transformer blocks per attention: {num_transformer_blocks}")

            if hasattr(block, 'downsamplers') and block.downsamplers and block.downsamplers[0] is not None:
                print(f"  ⬇️ Downsamples to: {res//2}x{res//2}")
                current_res = res // 2

            resolutions[block_name] = res

    # Mid block
    print("\n\n🎯 MIDDLE Block:")
    print("-" * 40)
    if hasattr(unet, 'mid_block'):
        print(f"\nmid_block:")
        print(f"  Resolution: {current_res}x{current_res} (bottleneck)")

        num_attentions = len(unet.mid_block.attentions) if hasattr(unet.mid_block, 'attentions') else 0
        num_resnets = len(unet.mid_block.resnets) if hasattr(unet.mid_block, 'resnets') else 0

        print(f"  ResNet layers: {num_resnets}")
        print(f"  Attention layers: {num_attentions}")

        if num_attentions > 0 and hasattr(unet.mid_block.attentions[0], 'transformer_blocks'):
            num_transformer_blocks = len(unet.mid_block.attentions[0].transformer_blocks)
            print(f"  Transformer blocks per attention: {num_transformer_blocks}")

        resolutions['mid_block'] = current_res

    # Up blocks
    print("\n\n📤 DECODER (Upsampling) Path:")
    print("-" * 40)

    up_block_configs = [
        (0, 32, True),    # Upsample to 64
        (1, 64, True),    # Upsample to 128
        (2, 128, False),  # No upsampling
    ]

    for block_idx, (idx, res, upsample) in enumerate(up_block_configs):
        if hasattr(unet, 'up_blocks'):
            block = unet.up_blocks[idx]
            block_name = f"up_blocks.{idx}"

            if upsample and hasattr(block, 'upsamplers') and block.upsamplers and block.upsamplers[0] is not None:
                current_res = res * 2
                print(f"\n{block_name}:")
                print(f"  ⬆️ Upsamples from {res}x{res} to {current_res}x{current_res}")
            else:
                print(f"\n{block_name}:")
                print(f"  Resolution: {res}x{res}")

            # Count attention layers
            num_attentions = len(block.attentions) if hasattr(block, 'attentions') else 0
            num_resnets = len(block.resnets) if hasattr(block, 'resnets') else 0

            print(f"  ResNet layers: {num_resnets}")
            print(f"  Attention layers: {num_attentions}")

            if num_attentions > 0 and hasattr(block.attentions[0], 'transformer_blocks'):
                num_transformer_blocks = len(block.attentions[0].transformer_blocks)
                print(f"  Transformer blocks per attention: {num_transformer_blocks}")

            resolutions[block_name] = res if not upsample else current_res

    print("\n\n💠 OUTPUT:")
    print("-" * 40)
    print(f"Final latent: 128x128 (4 channels)")
    print(f"After VAE decode: 1024x1024 (3 channels RGB)")

    return resolutions

def main():
    print("Loading SDXL model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    )

    unet = pipe.unet

    # Count all attention blocks
    print("\n" + "="*60)
    print("COMPLETE ATTENTION BLOCK INVENTORY")
    print("="*60)

    attention_blocks = {}

    # Down blocks
    for i, block in enumerate(unet.down_blocks):
        explore_block_attention_layers(block, f"down_blocks.{i}", attention_blocks)

    # Mid block
    if hasattr(unet, 'mid_block'):
        explore_block_attention_layers(unet.mid_block, "mid_block", attention_blocks)

    # Up blocks
    for i, block in enumerate(unet.up_blocks):
        explore_block_attention_layers(block, f"up_blocks.{i}", attention_blocks)

    # Print summary
    print(f"\n📊 TOTAL ATTENTION BLOCKS: {len(attention_blocks)}")
    print("\nBreakdown by section:")

    down_blocks = [k for k in attention_blocks if k.startswith('down_blocks')]
    mid_blocks = [k for k in attention_blocks if k.startswith('mid_block')]
    up_blocks = [k for k in attention_blocks if k.startswith('up_blocks')]

    print(f"  Down blocks: {len(down_blocks)} attention layers")
    print(f"  Mid block:   {len(mid_blocks)} attention layers")
    print(f"  Up blocks:   {len(up_blocks)} attention layers")

    # List all blocks
    print("\n" + "="*60)
    print("ALL ATTENTION BLOCKS:")
    print("="*60)

    for section, blocks in [("DOWN BLOCKS", down_blocks),
                            ("MID BLOCK", mid_blocks),
                            ("UP BLOCKS", up_blocks)]:
        print(f"\n{section}:")
        for block in sorted(blocks):
            info = attention_blocks[block]
            if 'q_shape' in info:
                print(f"  {block}")
                print(f"    Shape: {info['q_shape']}")

    # Trace resolution flow
    resolutions = trace_resolution_flow(unet)

    # Answer specific questions
    print("\n" + "="*60)
    print("ANSWERS TO YOUR QUESTIONS:")
    print("="*60)

    print(f"\n1. Total attention blocks in SDXL: {len(attention_blocks)}")
    print(f"   - Each has attn1 (self-attention) and often attn2 (cross-attention)")
    print(f"   - Each attention has to_q, to_k, to_v projections")

    print(f"\n2. The 128x128 'finished' latent:")
    print(f"   - INPUT: 128x128 latent (from VAE encode of 1024x1024 image)")
    print(f"   - Goes through downsampling: 128 → 64 → 32")
    print(f"   - Bottleneck at mid_block: 32x32")
    print(f"   - Goes through upsampling: 32 → 64 → 128")
    print(f"   - OUTPUT: 128x128 latent (ready for VAE decode)")
    print(f"   - VAE decodes each latent 'pixel' to 8x8 RGB pixels")

    print("\n📝 Note: The 128x128 latent is both the input AND output of the UNet!")
    print("   The UNet modifies this latent to denoise/generate the image.")

if __name__ == "__main__":
    main()