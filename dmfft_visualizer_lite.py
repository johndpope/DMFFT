"""
DMFFT SDXL Architecture Visualizer - Lite Version
===================================================

Uses file paths instead of base64 for faster loading.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class BlockCapture:
    block_name: str
    block_type: str
    channels: int
    block_idx: int
    timestep: float
    backbone_path: Optional[str] = None
    backbone_fft_path: Optional[str] = None
    skip_path: Optional[str] = None
    skip_fft_path: Optional[str] = None


def parse_debug_directory(debug_dir: str) -> Tuple[Dict[str, List[BlockCapture]], Optional[str]]:
    debug_path = Path(debug_dir).resolve()
    if not debug_path.exists():
        raise ValueError(f"Debug directory not found: {debug_dir}")

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

        match = pattern.match(f.name)
        if match:
            block_type, channels, block_idx, timestep, img_type = match.groups()
            key = f"{block_type}_{channels}ch_block{block_idx}"

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

            if img_type == 'backbone':
                capture.backbone_path = str(f)
            elif img_type == 'backbone_fft':
                capture.backbone_fft_path = str(f)
            elif img_type == 'skip':
                capture.skip_path = str(f)
            elif img_type == 'skip_fft':
                capture.skip_fft_path = str(f)
            continue

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
                    channels=320,
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

    for key in captures:
        captures[key].sort(key=lambda x: x.timestep, reverse=True)

    return dict(captures), latent_gif


def generate_lite_html(debug_dir: str, output_path: str = "dmfft_lite.html") -> str:
    captures, latent_gif = parse_debug_directory(debug_dir)
    debug_path = Path(debug_dir).resolve()

    all_timesteps = set()
    for block_captures in captures.values():
        for c in block_captures:
            all_timesteps.add(c.timestep)
    timesteps = sorted(all_timesteps, reverse=True)

    def block_sort_key(name):
        if '1280ch' in name:
            priority = 0
        elif '640ch' in name:
            priority = 1
        else:
            priority = 2
        match = re.search(r'block(\d+)', name)
        block_num = int(match.group(1)) if match else 0
        return (priority, block_num)

    available_blocks = sorted(captures.keys(), key=block_sort_key)

    blocks_info = []
    for block_name in available_blocks:
        block_captures = captures[block_name]
        if not block_captures:
            continue

        sample = block_captures[0]

        timestep_images = {}
        for cap in block_captures:
            t = str(cap.timestep)
            # Use relative paths for HTTP server
            timestep_images[t] = {
                'backbone': os.path.relpath(cap.backbone_path) if cap.backbone_path else None,
                'backbone_fft': os.path.relpath(cap.backbone_fft_path) if cap.backbone_fft_path else None,
                'skip': os.path.relpath(cap.skip_path) if cap.skip_path else None,
                'skip_fft': os.path.relpath(cap.skip_fft_path) if cap.skip_fft_path else None,
            }

        blocks_info.append({
            'name': block_name,
            'type': sample.block_type,
            'channels': sample.channels,
            'block_idx': sample.block_idx,
            'timesteps': [str(t) for t in sorted([float(t) for t in timestep_images.keys()], reverse=True)],
            'images': timestep_images
        })

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DMFFT SDXL Visualizer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 50%, #0a1a2a 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', system-ui, sans-serif;
            color: #e0e0e0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            background: linear-gradient(90deg, #00d4ff, #ff00d4, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }}
        .controls {{
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .slider {{ width: 200px; }}
        .btn {{
            background: rgba(0, 212, 255, 0.2);
            border: 1px solid #00d4ff;
            color: #00d4ff;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
        }}
        .btn:hover {{ background: rgba(0, 212, 255, 0.4); }}
        .btn.active {{ background: rgba(0, 212, 255, 0.5); }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 15px;
        }}
        .card {{
            background: rgba(0,0,0,0.4);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card:hover {{
            border-color: #00d4ff;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }}
        .card-header {{
            padding: 10px 15px;
            display: flex;
            justify-content: space-between;
        }}
        .card-header.crossattn {{ background: linear-gradient(90deg, rgba(255,100,100,0.3), rgba(255,150,50,0.3)); border-bottom: 2px solid #ff6464; }}
        .card-header.upblock {{ background: linear-gradient(90deg, rgba(100,200,100,0.3), rgba(50,255,150,0.3)); border-bottom: 2px solid #64ff64; }}
        .card-images {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2px;
            padding: 2px;
        }}
        .card-images img {{
            width: 100%;
            height: auto;
            background: #111;
        }}
        .img-label {{
            font-size: 0.7em;
            color: #888;
            text-align: center;
            padding: 2px;
            background: rgba(0,0,0,0.5);
        }}
        .latent-section {{
            margin-top: 20px;
            text-align: center;
        }}
        .latent-section img {{
            max-width: 400px;
            border-radius: 8px;
        }}
        .stats {{ font-size: 0.8em; color: #888; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>DMFFT SDXL Architecture Visualizer</h1>

    <div class="controls">
        <label>Timestep: <input type="range" class="slider" id="slider" min="0" max="{len(timesteps)-1}" value="0"></label>
        <span id="tDisplay">t={timesteps[0] if timesteps else 0}</span>
        <button class="btn" id="playBtn" onclick="togglePlay()">▶ Play</button>
        <div>
            <button class="btn active" onclick="setType('backbone')">Backbone</button>
            <button class="btn" onclick="setType('skip')">Skip</button>
            <button class="btn" onclick="setType('backbone_fft')">FFT</button>
            <button class="btn" onclick="setType('skip_fft')">Skip FFT</button>
        </div>
    </div>

    <div class="grid" id="grid"></div>

    <div class="latent-section">
        <h3 style="color: #888; margin-bottom: 10px;">Latent Evolution</h3>
        <img src="{os.path.relpath(latent_gif) if latent_gif else ''}" alt="Latent Evolution" onerror="this.style.display='none'">
    </div>

    <script>
        const blocks = {json.dumps(blocks_info)};
        const timesteps = {json.dumps([str(t) for t in timesteps])};
        let currentT = timesteps[0];
        let imageType = 'backbone';
        let playing = false;
        let interval;

        function render() {{
            const grid = document.getElementById('grid');
            grid.innerHTML = '';

            blocks.forEach(block => {{
                const images = block.images[currentT] || {{}};
                const headerClass = block.type.includes('CrossAttn') ? 'crossattn' : 'upblock';

                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <div class="card-header ${{headerClass}}">
                        <span>${{block.name.replace(/_/g, ' ')}}</span>
                        <span style="color: #888">${{block.channels}}ch</span>
                    </div>
                    <div class="card-images">
                        <div>
                            ${{images.backbone ? `<img src="${{images.backbone}}" alt="backbone">` : '<div style="height:100px;background:#222"></div>'}}
                            <div class="img-label">Backbone</div>
                        </div>
                        <div>
                            ${{images.skip ? `<img src="${{images.skip}}" alt="skip">` : '<div style="height:100px;background:#222"></div>'}}
                            <div class="img-label">Skip</div>
                        </div>
                        <div>
                            ${{images.backbone_fft ? `<img src="${{images.backbone_fft}}" alt="fft">` : '<div style="height:100px;background:#222"></div>'}}
                            <div class="img-label">Backbone FFT</div>
                        </div>
                        <div>
                            ${{images.skip_fft ? `<img src="${{images.skip_fft}}" alt="skip_fft">` : '<div style="height:100px;background:#222"></div>'}}
                            <div class="img-label">Skip FFT</div>
                        </div>
                    </div>
                    <div class="stats">Block ${{block.block_idx}} | ${{block.timesteps.length}} timesteps</div>
                `;
                grid.appendChild(card);
            }});
        }}

        document.getElementById('slider').addEventListener('input', (e) => {{
            currentT = timesteps[e.target.value];
            document.getElementById('tDisplay').textContent = 't=' + currentT;
            render();
        }});

        function setType(t) {{
            imageType = t;
            document.querySelectorAll('.controls .btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            render();
        }}

        function togglePlay() {{
            playing = !playing;
            const btn = document.getElementById('playBtn');
            if (playing) {{
                btn.textContent = '⏸ Pause';
                btn.classList.add('active');
                interval = setInterval(() => {{
                    const slider = document.getElementById('slider');
                    slider.value = (parseInt(slider.value) + 1) % timesteps.length;
                    currentT = timesteps[slider.value];
                    document.getElementById('tDisplay').textContent = 't=' + currentT;
                    render();
                }}, 500);
            }} else {{
                btn.textContent = '▶ Play';
                btn.classList.remove('active');
                clearInterval(interval);
            }}
        }}

        render();
    </script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Lite visualization saved to: {output_path}")
    print(f"Found {len(blocks_info)} blocks with {len(timesteps)} timesteps each")
    return output_path


if __name__ == '__main__':
    import sys
    debug_dir = sys.argv[1] if len(sys.argv) > 1 else 'debug_output/gen_0002'
    output = generate_lite_html(debug_dir)
    os.system(f"xdg-open {output} 2>/dev/null &")
