"""
DMFFT SDXL Architecture Visualizer
===================================

Creates an interactive HTML visualization showing:
- Complete SDXL UNet architecture with DMFFT modifications
- All captured feature maps, FFT analysis, skip connections
- Timestep-based animation of the denoising process
- Isometric 3D view with image galleries
"""

import os
import re
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class BlockCapture:
    """Represents captured data for a single block at a timestep."""
    block_name: str
    block_type: str  # CrossAttnUpBlock2D, UpBlock2D, etc.
    channels: int
    block_idx: int
    timestep: float
    backbone_path: Optional[str] = None
    backbone_fft_path: Optional[str] = None
    skip_path: Optional[str] = None
    skip_fft_path: Optional[str] = None


def image_to_base64(path: str) -> str:
    """Convert image file to base64 data URI."""
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    ext = Path(path).suffix.lower()
    mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'gif': 'image/gif'}.get(ext[1:], 'image/png')
    return f"data:{mime};base64,{data}"


def parse_debug_directory(debug_dir: str) -> Tuple[Dict[str, List[BlockCapture]], Optional[str]]:
    """
    Parse debug output directory and organize captures by block and timestep.

    Returns:
        Tuple of (blocks_dict, latent_gif_path)
    """
    debug_path = Path(debug_dir)
    if not debug_path.exists():
        raise ValueError(f"Debug directory not found: {debug_dir}")

    # Pattern: BlockType_ChannelsCh_blockN_tTimestep_type.png
    pattern = re.compile(r'^(.+?)_(\d+)ch_block(\d+)_t([\d.]+)_(.+)\.png$')
    pattern_upblock = re.compile(r'^(UpBlock2D)_block(\d+)_t([\d.]+)_(.+)\.png$')

    captures = defaultdict(list)
    latent_gif = None

    for f in debug_path.iterdir():
        if f.name == 'latent_evolution.gif':
            latent_gif = str(f)
            continue

        if not f.suffix == '.png':
            continue

        # Try pattern with channels
        match = pattern.match(f.name)
        if match:
            block_type, channels, block_idx, timestep, img_type = match.groups()
            key = f"{block_type}_{channels}ch_block{block_idx}"

            # Find or create capture for this timestep
            capture = None
            for c in captures[key]:
                if c.timestep == float(timestep):
                    capture = c
                    break

            if capture is None:
                capture = BlockCapture(
                    block_name=key,
                    block_type=block_type,
                    channels=int(channels),
                    block_idx=int(block_idx),
                    timestep=float(timestep)
                )
                captures[key].append(capture)

            # Set the appropriate path
            if img_type == 'backbone':
                capture.backbone_path = str(f)
            elif img_type == 'backbone_fft':
                capture.backbone_fft_path = str(f)
            elif img_type == 'skip':
                capture.skip_path = str(f)
            elif img_type == 'skip_fft':
                capture.skip_fft_path = str(f)
            continue

        # Try pattern without channels (UpBlock2D)
        match = pattern_upblock.match(f.name)
        if match:
            block_type, block_idx, timestep, img_type = match.groups()
            key = f"{block_type}_block{block_idx}"

            capture = None
            for c in captures[key]:
                if c.timestep == float(timestep):
                    capture = c
                    break

            if capture is None:
                capture = BlockCapture(
                    block_name=key,
                    block_type=block_type,
                    channels=320,  # Default for UpBlock2D
                    block_idx=int(block_idx),
                    timestep=float(timestep)
                )
                captures[key].append(capture)

            if img_type == 'backbone':
                capture.backbone_path = str(f)
            elif img_type == 'backbone_fft':
                capture.backbone_fft_path = str(f)
            elif img_type == 'skip':
                capture.skip_path = str(f)
            elif img_type == 'skip_fft':
                capture.skip_fft_path = str(f)

    # Sort captures by timestep
    for key in captures:
        captures[key].sort(key=lambda x: x.timestep, reverse=True)  # High timestep first (noisy -> clean)

    return dict(captures), latent_gif


def generate_dmfft_html(debug_dir: str, output_path: str = "dmfft_viz.html") -> str:
    """Generate comprehensive DMFFT visualization HTML."""

    captures, latent_gif = parse_debug_directory(debug_dir)

    # Get all unique timesteps
    all_timesteps = set()
    for block_captures in captures.values():
        for c in block_captures:
            all_timesteps.add(c.timestep)
    timesteps = sorted(all_timesteps, reverse=True)

    # Organize blocks - sort by type and block index
    def block_sort_key(name):
        # CrossAttn 1280 first, then 640, then UpBlock
        if '1280ch' in name:
            priority = 0
        elif '640ch' in name:
            priority = 1
        else:
            priority = 2
        # Extract block number
        import re
        match = re.search(r'block(\d+)', name)
        block_num = int(match.group(1)) if match else 0
        return (priority, block_num)

    available_blocks = sorted(captures.keys(), key=block_sort_key)

    # Build block info for JavaScript
    blocks_info = []
    for block_name in available_blocks:
        block_captures = captures[block_name]
        if not block_captures:
            continue

        sample = block_captures[0]

        # Collect all images for this block (use string keys for JS compatibility)
        timestep_images = {}
        for cap in block_captures:
            t = str(cap.timestep)  # Convert to string for JS object keys
            timestep_images[t] = {
                'backbone': image_to_base64(cap.backbone_path) if cap.backbone_path else None,
                'backbone_fft': image_to_base64(cap.backbone_fft_path) if cap.backbone_fft_path else None,
                'skip': image_to_base64(cap.skip_path) if cap.skip_path else None,
                'skip_fft': image_to_base64(cap.skip_fft_path) if cap.skip_fft_path else None,
            }

        blocks_info.append({
            'name': block_name,
            'type': sample.block_type,
            'channels': sample.channels,
            'block_idx': sample.block_idx,
            'timesteps': list(timestep_images.keys()),
            'images': timestep_images
        })

    # Latent evolution GIF
    latent_gif_data = image_to_base64(latent_gif) if latent_gif else None

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DMFFT SDXL Architecture Visualizer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 50%, #0a1a2a 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', system-ui, sans-serif;
            color: #e0e0e0;
        }}

        .header {{
            background: rgba(0,0,0,0.5);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid rgba(100, 200, 255, 0.3);
        }}

        .header h1 {{
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #ff00d4, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        }}

        .header p {{
            color: #888;
            margin-top: 5px;
        }}

        .controls {{
            background: rgba(0,0,0,0.3);
            padding: 15px 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .control-group label {{
            color: #aaa;
            font-size: 0.9em;
        }}

        .slider {{
            -webkit-appearance: none;
            width: 300px;
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bff6b);
            outline: none;
        }}

        .slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #fff;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(255,255,255,0.5);
        }}

        .timestep-display {{
            font-family: monospace;
            font-size: 1.2em;
            color: #00d4ff;
            min-width: 80px;
        }}

        .btn {{
            background: rgba(0, 212, 255, 0.2);
            border: 1px solid #00d4ff;
            color: #00d4ff;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
        }}

        .btn:hover {{
            background: rgba(0, 212, 255, 0.4);
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
        }}

        .btn.active {{
            background: rgba(0, 212, 255, 0.5);
        }}

        .main-container {{
            display: flex;
            height: calc(100vh - 150px);
        }}

        .architecture-panel {{
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }}

        .arch-title {{
            text-align: center;
            margin-bottom: 20px;
            color: #00d4ff;
            font-size: 1.3em;
        }}

        .block-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
            padding: 10px;
        }}

        .block-card {{
            background: rgba(0,0,0,0.4);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.3s;
            cursor: pointer;
        }}

        .block-card:hover {{
            transform: translateY(-5px);
            border-color: rgba(0, 212, 255, 0.5);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
        }}

        .block-card.selected {{
            border-color: #00d4ff;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
        }}

        .block-header {{
            padding: 12px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .block-header.crossattn {{
            background: linear-gradient(90deg, rgba(255, 100, 100, 0.3), rgba(255, 150, 50, 0.3));
            border-bottom: 2px solid #ff6464;
        }}

        .block-header.upblock {{
            background: linear-gradient(90deg, rgba(100, 200, 100, 0.3), rgba(50, 255, 150, 0.3));
            border-bottom: 2px solid #64ff64;
        }}

        .block-name {{
            font-weight: bold;
            font-size: 0.95em;
        }}

        .block-meta {{
            font-size: 0.8em;
            color: #888;
        }}

        .block-images {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2px;
            padding: 2px;
            background: rgba(0,0,0,0.3);
        }}

        .block-images img {{
            width: 100%;
            height: auto;
            display: block;
        }}

        .image-label {{
            position: absolute;
            bottom: 2px;
            left: 2px;
            background: rgba(0,0,0,0.7);
            color: #fff;
            font-size: 0.7em;
            padding: 2px 5px;
            border-radius: 3px;
        }}

        .image-cell {{
            position: relative;
            background: #111;
        }}

        .detail-panel {{
            width: 450px;
            background: rgba(0,0,0,0.5);
            border-left: 1px solid rgba(255,255,255,0.1);
            padding: 20px;
            overflow-y: auto;
        }}

        .detail-title {{
            color: #00d4ff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .detail-section {{
            margin-bottom: 20px;
        }}

        .detail-section h3 {{
            color: #888;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}

        .detail-image {{
            width: 100%;
            border-radius: 8px;
            margin-bottom: 10px;
        }}

        .stat-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}

        .stat-box {{
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }}

        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #00d4ff;
        }}

        .stat-label {{
            font-size: 0.8em;
            color: #666;
        }}

        .latent-section {{
            margin-top: 20px;
            text-align: center;
        }}

        .latent-section img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.5);
        }}

        .flow-diagram {{
            margin: 20px 0;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.85em;
            line-height: 1.8;
            text-align: center;
        }}

        .flow-arrow {{
            color: #00d4ff;
        }}

        .view-toggle {{
            display: flex;
            gap: 5px;
        }}

        .tab-btn {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            color: #888;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s;
        }}

        .tab-btn:hover {{
            background: rgba(255,255,255,0.1);
        }}

        .tab-btn.active {{
            background: rgba(0, 212, 255, 0.2);
            border-color: #00d4ff;
            color: #00d4ff;
        }}

        .image-type-toggle {{
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }}

        @keyframes flow {{
            0% {{ background-position: 0% 50%; }}
            100% {{ background-position: 200% 50%; }}
        }}

        .flowing-border {{
            background: linear-gradient(90deg, #00d4ff, #ff00d4, #00ff88, #00d4ff);
            background-size: 200% 100%;
            animation: flow 3s linear infinite;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>DMFFT SDXL Architecture</h1>
        <p>Dual-Mode Fourier Feature Transform Visualization</p>
    </div>

    <div class="controls">
        <div class="control-group">
            <label>Timestep (Noise Level):</label>
            <input type="range" class="slider" id="timestepSlider" min="0" max="{len(timesteps)-1}" value="0">
            <span class="timestep-display" id="timestepDisplay">t={timesteps[0] if timesteps else 0}</span>
        </div>

        <div class="control-group">
            <button class="btn" id="playBtn" onclick="togglePlay()">▶ Play</button>
            <button class="btn" onclick="resetView()">↺ Reset</button>
        </div>

        <div class="view-toggle">
            <button class="tab-btn active" onclick="setImageType('backbone')">Backbone</button>
            <button class="tab-btn" onclick="setImageType('skip')">Skip</button>
            <button class="tab-btn" onclick="setImageType('backbone_fft')">Backbone FFT</button>
            <button class="tab-btn" onclick="setImageType('skip_fft')">Skip FFT</button>
        </div>
    </div>

    <div class="main-container">
        <div class="architecture-panel">
            <div class="arch-title">SDXL U-Net Decoder Path (Up Blocks)</div>

            <div class="flow-diagram">
                <span style="color: #ff6464;">Mid Block (32×32, 1280ch)</span>
                <span class="flow-arrow"> → </span>
                <span style="color: #ff9664;">CrossAttn 1280ch</span>
                <span class="flow-arrow"> → </span>
                <span style="color: #ffb464;">CrossAttn 640ch</span>
                <span class="flow-arrow"> → </span>
                <span style="color: #64ff64;">UpBlock 320ch</span>
                <span class="flow-arrow"> → </span>
                <span style="color: #00d4ff;">Output (128×128)</span>
            </div>

            <div class="block-grid" id="blockGrid">
                <!-- Blocks will be populated by JavaScript -->
            </div>
        </div>

        <div class="detail-panel">
            <h2 class="detail-title" id="detailTitle">Select a Block</h2>

            <div id="detailContent">
                <div class="flow-diagram">
                    Click on any block card to see detailed feature maps and FFT analysis.
                </div>

                <div class="latent-section">
                    <h3 style="color: #888; margin-bottom: 10px;">Latent Evolution</h3>
                    {"<img src='" + latent_gif_data + "' alt='Latent Evolution'>" if latent_gif_data else "<p>No latent GIF available</p>"}
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data from Python
        const blocksInfo = {json.dumps(blocks_info)};
        const timesteps = {json.dumps(timesteps)};

        // State
        let currentTimestep = timesteps[0] || 0;
        let currentImageType = 'backbone';
        let selectedBlock = null;
        let isPlaying = false;
        let playInterval = null;

        // Initialize
        function init() {{
            renderBlocks();
            updateTimestepDisplay();
        }}

        function renderBlocks() {{
            const grid = document.getElementById('blockGrid');
            grid.innerHTML = '';

            blocksInfo.forEach((block, idx) => {{
                const card = document.createElement('div');
                card.className = 'block-card' + (selectedBlock === block.name ? ' selected' : '');
                card.onclick = () => selectBlock(block.name);

                const headerClass = block.type.includes('CrossAttn') ? 'crossattn' : 'upblock';

                // Get image for current timestep (use string key)
                const images = block.images[String(currentTimestep)] || {{}};
                const imgSrc = images[currentImageType];

                card.innerHTML = `
                    <div class="block-header ${{headerClass}}">
                        <span class="block-name">${{block.name.replace(/_/g, ' ')}}</span>
                        <span class="block-meta">${{block.channels}}ch</span>
                    </div>
                    <div class="block-images">
                        ${{imgSrc ? `<div class="image-cell" style="grid-column: span 2;">
                            <img src="${{imgSrc}}" alt="${{currentImageType}}">
                            <span class="image-label">${{currentImageType}}</span>
                        </div>` : '<div style="grid-column: span 2; padding: 40px; text-align: center; color: #444;">No image</div>'}}
                    </div>
                `;

                grid.appendChild(card);
            }});
        }}

        function selectBlock(blockName) {{
            selectedBlock = blockName;
            renderBlocks();
            renderDetail();
        }}

        function renderDetail() {{
            if (!selectedBlock) return;

            const block = blocksInfo.find(b => b.name === selectedBlock);
            if (!block) return;

            const images = block.images[String(currentTimestep)] || {{}};

            document.getElementById('detailTitle').textContent = block.name.replace(/_/g, ' ');

            let html = `
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-value">${{block.channels}}</div>
                        <div class="stat-label">Channels</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">${{block.block_idx}}</div>
                        <div class="stat-label">Block Index</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">${{block.timesteps.length}}</div>
                        <div class="stat-label">Timesteps</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">${{currentTimestep.toFixed(0)}}</div>
                        <div class="stat-label">Current t</div>
                    </div>
                </div>
            `;

            // Show all image types for this block
            const imageTypes = [
                ['backbone', 'Backbone Features'],
                ['backbone_fft', 'Backbone FFT'],
                ['skip', 'Skip Connection'],
                ['skip_fft', 'Skip FFT']
            ];

            imageTypes.forEach(([key, label]) => {{
                if (images[key]) {{
                    html += `
                        <div class="detail-section">
                            <h3>${{label}}</h3>
                            <img src="${{images[key]}}" class="detail-image" alt="${{label}}">
                        </div>
                    `;
                }}
            }});

            document.getElementById('detailContent').innerHTML = html;
        }}

        function setImageType(type) {{
            currentImageType = type;

            // Update button states
            document.querySelectorAll('.view-toggle .tab-btn').forEach(btn => {{
                btn.classList.remove('active');
                if (btn.textContent.toLowerCase().replace(/ /g, '_').includes(type) ||
                    (type === 'backbone' && btn.textContent === 'Backbone') ||
                    (type === 'skip' && btn.textContent === 'Skip') ||
                    (type === 'backbone_fft' && btn.textContent === 'Backbone FFT') ||
                    (type === 'skip_fft' && btn.textContent === 'Skip FFT')) {{
                    btn.classList.add('active');
                }}
            }});

            renderBlocks();
        }}

        function updateTimestepDisplay() {{
            const slider = document.getElementById('timestepSlider');
            const idx = parseInt(slider.value);
            currentTimestep = timesteps[idx] || 0;
            document.getElementById('timestepDisplay').textContent = `t=${{currentTimestep.toFixed(0)}}`;
            renderBlocks();
            if (selectedBlock) renderDetail();
        }}

        document.getElementById('timestepSlider').addEventListener('input', updateTimestepDisplay);

        function togglePlay() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('playBtn');

            if (isPlaying) {{
                btn.textContent = '⏸ Pause';
                btn.classList.add('active');
                playInterval = setInterval(() => {{
                    const slider = document.getElementById('timestepSlider');
                    let idx = parseInt(slider.value);
                    idx = (idx + 1) % timesteps.length;
                    slider.value = idx;
                    updateTimestepDisplay();
                }}, 500);
            }} else {{
                btn.textContent = '▶ Play';
                btn.classList.remove('active');
                clearInterval(playInterval);
            }}
        }}

        function resetView() {{
            selectedBlock = null;
            document.getElementById('timestepSlider').value = 0;
            updateTimestepDisplay();
            document.getElementById('detailTitle').textContent = 'Select a Block';
            document.getElementById('detailContent').innerHTML = `
                <div class="flow-diagram">
                    Click on any block card to see detailed feature maps and FFT analysis.
                </div>
            `;
            renderBlocks();
        }}

        // Initialize
        init();
    </script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"DMFFT visualization saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        debug_dir = sys.argv[1]
    else:
        debug_dir = 'debug_output/gen_0002'

    output = generate_dmfft_html(debug_dir)
    print(f"Opening {output}...")
    os.system(f"xdg-open {output} 2>/dev/null &")
