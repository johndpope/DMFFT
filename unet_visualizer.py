"""
Interactive HTML Canvas UNet Visualizer
========================================

Creates an isometric 3D visualization of SDXL UNet blocks
with animated data flow, hover effects, and feature map previews.
"""

import json
import base64
import io
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm


def tensor_to_colormap_base64(
    tensor: torch.Tensor,
    colormap: str = 'viridis',
    size: Tuple[int, int] = (64, 64)
) -> str:
    """Convert tensor to base64 encoded colormap image."""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch
    if tensor.dim() == 3:
        # Average across channels or take first few
        tensor = tensor[:3].mean(dim=0) if tensor.shape[0] >= 3 else tensor[0]

    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(arr)[:, :, :3]  # RGB only
    colored = (colored * 255).astype(np.uint8)

    # Resize
    img = Image.fromarray(colored)
    img = img.resize(size, Image.LANCZOS)

    # To base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_multi_channel_preview(
    tensor: torch.Tensor,
    n_channels: int = 4,
    colormap: str = 'plasma'
) -> str:
    """Create a 2x2 grid preview of multiple channels."""
    if tensor.dim() == 4:
        tensor = tensor[0]

    c, h, w = tensor.shape
    indices = torch.linspace(0, c-1, n_channels).long()

    # Create 2x2 grid
    grid_size = 64
    grid = np.zeros((grid_size * 2, grid_size * 2, 3), dtype=np.uint8)

    cmap = plt.get_cmap(colormap)

    for i, idx in enumerate(indices):
        ch = tensor[idx].detach().cpu().numpy()
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        colored = (cmap(ch)[:, :, :3] * 255).astype(np.uint8)

        img = Image.fromarray(colored).resize((grid_size, grid_size), Image.LANCZOS)
        row, col = i // 2, i % 2
        grid[row*grid_size:(row+1)*grid_size, col*grid_size:(col+1)*grid_size] = np.array(img)

    buffer = io.BytesIO()
    Image.fromarray(grid).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class UNetVisualizer:
    """Generate interactive HTML visualization of UNet."""

    # Color schemes for different block types
    COLORS = {
        'down': {'fill': '#4a90d9', 'stroke': '#2d5a8a', 'glow': '#6bb3ff'},
        'mid': {'fill': '#d94a4a', 'stroke': '#8a2d2d', 'glow': '#ff6b6b'},
        'up': {'fill': '#4ad94a', 'stroke': '#2d8a2d', 'glow': '#6bff6b'},
        'skip': {'stroke': '#ffa500', 'glow': '#ffcc00'},
    }

    # Colormaps for each block type
    COLORMAPS = {
        'down': 'viridis',
        'mid': 'magma',
        'up': 'plasma',
    }

    def __init__(self):
        self.blocks_data: List[Dict] = []
        self.connections: List[Dict] = []
        self.feature_previews: Dict[str, str] = {}

    def add_block(
        self,
        name: str,
        block_type: str,  # 'down', 'mid', 'up'
        resolution: int,
        channels: int,
        features: Optional[torch.Tensor] = None,
        attention_count: int = 0,
        transformer_blocks: int = 0
    ):
        """Add a UNet block to the visualization."""
        block = {
            'name': name,
            'type': block_type,
            'resolution': resolution,
            'channels': channels,
            'attention_count': attention_count,
            'transformer_blocks': transformer_blocks,
            'preview': None,
            'stats': {}
        }

        if features is not None:
            colormap = self.COLORMAPS.get(block_type, 'viridis')
            block['preview'] = create_multi_channel_preview(features, colormap=colormap)

            # Compute stats
            with torch.no_grad():
                block['stats'] = {
                    'mean': float(features.mean()),
                    'std': float(features.std()),
                    'min': float(features.min()),
                    'max': float(features.max()),
                }

        self.blocks_data.append(block)

    def add_skip_connection(self, from_block: str, to_block: str):
        """Add a skip connection between blocks."""
        self.connections.append({
            'from': from_block,
            'to': to_block,
            'type': 'skip'
        })

    def generate_html(self, output_path: str = "unet_viz.html", title: str = "SDXL UNet Visualizer"):
        """Generate the interactive HTML visualization."""

        blocks_json = json.dumps(self.blocks_data)
        connections_json = json.dumps(self.connections)
        colors_json = json.dumps(self.COLORS)

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', system-ui, sans-serif;
            color: #e0e0e0;
            overflow-x: hidden;
        }}

        .header {{
            text-align: center;
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .header h1 {{
            font-size: 2em;
            background: linear-gradient(90deg, #4a90d9, #d94a4a, #4ad94a);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .container {{
            display: flex;
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }}

        .canvas-container {{
            flex: 1;
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            padding: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        #unetCanvas {{
            display: block;
            margin: 0 auto;
        }}

        .info-panel {{
            width: 350px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        .info-panel h2 {{
            color: #4a90d9;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .block-info {{
            margin-bottom: 20px;
        }}

        .block-info .title {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}

        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}

        .stat-label {{
            color: #888;
        }}

        .stat-value {{
            font-family: monospace;
            color: #4ad94a;
        }}

        .preview-container {{
            margin-top: 15px;
            text-align: center;
        }}

        .preview-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}

        .legend {{
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
        }}

        .controls {{
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}

        .control-btn {{
            background: rgba(74, 144, 217, 0.3);
            border: 1px solid #4a90d9;
            color: #4a90d9;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            margin: 5px;
            transition: all 0.3s;
        }}

        .control-btn:hover {{
            background: rgba(74, 144, 217, 0.5);
        }}

        .flow-indicator {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 0.9em;
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 0.6; }}
            50% {{ opacity: 1; }}
        }}

        .animating {{
            animation: pulse 1s infinite;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SDXL U-Net Architecture</h1>
        <p>Interactive Isometric Visualization</p>
    </div>

    <div class="container">
        <div class="canvas-container">
            <canvas id="unetCanvas" width="1000" height="700"></canvas>
        </div>

        <div class="info-panel">
            <h2>Block Inspector</h2>
            <div id="blockInfo" class="block-info">
                <p style="color: #666;">Hover over a block to see details</p>
            </div>

            <div class="legend">
                <h3 style="margin-bottom: 10px;">Legend</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4a90d9;"></div>
                    <span>Down Blocks (Encoder)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #d94a4a;"></div>
                    <span>Mid Block (Bottleneck)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4ad94a;"></div>
                    <span>Up Blocks (Decoder)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: linear-gradient(90deg, #ffa500, #ffcc00);"></div>
                    <span>Skip Connections</span>
                </div>
            </div>

            <div class="controls">
                <h3 style="margin-bottom: 10px;">Controls</h3>
                <button class="control-btn" onclick="toggleAnimation()">Toggle Flow Animation</button>
                <button class="control-btn" onclick="toggleLabels()">Toggle Labels</button>
                <button class="control-btn" onclick="resetView()">Reset View</button>
            </div>
        </div>
    </div>

    <div class="flow-indicator" id="flowIndicator">
        Data Flow: Input Latent → Down → Mid → Up → Output Latent
    </div>

    <script>
        // Data from Python
        const blocksData = {blocks_json};
        const connections = {connections_json};
        const colors = {colors_json};

        // Canvas setup
        const canvas = document.getElementById('unetCanvas');
        const ctx = canvas.getContext('2d');

        // State
        let animationEnabled = true;
        let labelsEnabled = true;
        let hoveredBlock = null;
        let animationPhase = 0;
        let particles = [];

        // Isometric settings
        const ISO = {{
            angle: Math.PI / 6,  // 30 degrees
            scale: 1,
            offsetX: canvas.width / 2,
            offsetY: 100
        }};

        // Block visual settings
        const BLOCK_WIDTH = 80;
        const BLOCK_DEPTH = 60;
        const BLOCK_BASE_HEIGHT = 40;
        const VERTICAL_SPACING = 90;
        const HORIZONTAL_SPACING = 180;

        // Convert 3D to 2D isometric
        function isoProject(x, y, z) {{
            const isoX = (x - z) * Math.cos(ISO.angle) * ISO.scale + ISO.offsetX;
            const isoY = y + (x + z) * Math.sin(ISO.angle) * ISO.scale + ISO.offsetY;
            return {{ x: isoX, y: isoY }};
        }}

        // Draw isometric box
        function drawIsoBox(x, y, z, width, depth, height, fillColor, strokeColor, glowColor, isHovered) {{
            const corners = {{
                // Top face
                topFrontLeft: isoProject(x, y - height, z),
                topFrontRight: isoProject(x + width, y - height, z),
                topBackRight: isoProject(x + width, y - height, z + depth),
                topBackLeft: isoProject(x, y - height, z + depth),
                // Bottom face
                botFrontLeft: isoProject(x, y, z),
                botFrontRight: isoProject(x + width, y, z),
                botBackRight: isoProject(x + width, y, z + depth),
                botBackLeft: isoProject(x, y, z + depth),
            }};

            // Glow effect on hover
            if (isHovered) {{
                ctx.shadowColor = glowColor;
                ctx.shadowBlur = 20;
            }}

            // Right face
            ctx.beginPath();
            ctx.moveTo(corners.topFrontRight.x, corners.topFrontRight.y);
            ctx.lineTo(corners.topBackRight.x, corners.topBackRight.y);
            ctx.lineTo(corners.botBackRight.x, corners.botBackRight.y);
            ctx.lineTo(corners.botFrontRight.x, corners.botFrontRight.y);
            ctx.closePath();
            ctx.fillStyle = shadeColor(fillColor, -20);
            ctx.fill();
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 2;
            ctx.stroke();

            // Left face
            ctx.beginPath();
            ctx.moveTo(corners.topFrontLeft.x, corners.topFrontLeft.y);
            ctx.lineTo(corners.topBackLeft.x, corners.topBackLeft.y);
            ctx.lineTo(corners.botBackLeft.x, corners.botBackLeft.y);
            ctx.lineTo(corners.botFrontLeft.x, corners.botFrontLeft.y);
            ctx.closePath();
            ctx.fillStyle = shadeColor(fillColor, -40);
            ctx.fill();
            ctx.stroke();

            // Top face
            ctx.beginPath();
            ctx.moveTo(corners.topFrontLeft.x, corners.topFrontLeft.y);
            ctx.lineTo(corners.topFrontRight.x, corners.topFrontRight.y);
            ctx.lineTo(corners.topBackRight.x, corners.topBackRight.y);
            ctx.lineTo(corners.topBackLeft.x, corners.topBackLeft.y);
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.stroke();

            ctx.shadowBlur = 0;

            return corners;
        }}

        // Shade color helper
        function shadeColor(color, percent) {{
            const num = parseInt(color.replace('#', ''), 16);
            const amt = Math.round(2.55 * percent);
            const R = Math.max(0, Math.min(255, (num >> 16) + amt));
            const G = Math.max(0, Math.min(255, ((num >> 8) & 0x00FF) + amt));
            const B = Math.max(0, Math.min(255, (num & 0x0000FF) + amt));
            return '#' + (0x1000000 + R * 0x10000 + G * 0x100 + B).toString(16).slice(1);
        }}

        // Draw arrow
        function drawArrow(fromX, fromY, toX, toY, color, animated = false) {{
            const headLength = 10;
            const dx = toX - fromX;
            const dy = toY - fromY;
            const angle = Math.atan2(dy, dx);

            // Animated dash offset
            if (animated && animationEnabled) {{
                ctx.setLineDash([5, 5]);
                ctx.lineDashOffset = -animationPhase * 2;
            }} else {{
                ctx.setLineDash([]);
            }}

            ctx.beginPath();
            ctx.moveTo(fromX, fromY);
            ctx.lineTo(toX, toY);
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.stroke();

            // Arrowhead
            ctx.beginPath();
            ctx.moveTo(toX, toY);
            ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI / 6), toY - headLength * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI / 6), toY - headLength * Math.sin(angle + Math.PI / 6));
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();

            ctx.setLineDash([]);
        }}

        // Draw curved skip connection
        function drawSkipConnection(fromCorners, toCorners, animated = false) {{
            const startX = fromCorners.topBackLeft.x;
            const startY = fromCorners.topBackLeft.y;
            const endX = toCorners.topBackLeft.x;
            const endY = toCorners.topBackLeft.y;

            const midX = Math.min(startX, endX) - 50;
            const midY = (startY + endY) / 2;

            if (animated && animationEnabled) {{
                ctx.setLineDash([8, 4]);
                ctx.lineDashOffset = -animationPhase * 3;
            }}

            // Gradient for skip connection
            const gradient = ctx.createLinearGradient(startX, startY, endX, endY);
            gradient.addColorStop(0, '#ffa500');
            gradient.addColorStop(1, '#ffcc00');

            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.bezierCurveTo(midX, startY, midX, endY, endX, endY);
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.setLineDash([]);
        }}

        // Particle system for data flow
        class Particle {{
            constructor(path, color) {{
                this.path = path;
                this.progress = Math.random();
                this.speed = 0.005 + Math.random() * 0.005;
                this.color = color;
                this.size = 3 + Math.random() * 3;
            }}

            update() {{
                this.progress += this.speed;
                if (this.progress > 1) this.progress = 0;
            }}

            draw() {{
                const idx = Math.floor(this.progress * (this.path.length - 1));
                const nextIdx = Math.min(idx + 1, this.path.length - 1);
                const t = (this.progress * (this.path.length - 1)) - idx;

                const x = this.path[idx].x + (this.path[nextIdx].x - this.path[idx].x) * t;
                const y = this.path[idx].y + (this.path[nextIdx].y - this.path[idx].y) * t;

                ctx.beginPath();
                ctx.arc(x, y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = this.color;
                ctx.shadowColor = this.color;
                ctx.shadowBlur = 10;
                ctx.fill();
                ctx.shadowBlur = 0;
            }}
        }}

        // Prepare block positions
        const blockPositions = new Map();
        const blockCorners = new Map();

        function calculateBlockPositions() {{
            let downY = 0;
            let upY = 0;

            blocksData.forEach((block, index) => {{
                let x, y, z;
                const heightScale = Math.max(0.5, Math.min(2, block.channels / 1000));
                const height = BLOCK_BASE_HEIGHT * heightScale;

                if (block.type === 'down') {{
                    x = -HORIZONTAL_SPACING;
                    y = downY;
                    z = 0;
                    downY += VERTICAL_SPACING;
                }} else if (block.type === 'mid') {{
                    x = 0;
                    y = downY;
                    z = BLOCK_DEPTH;
                }} else if (block.type === 'up') {{
                    x = HORIZONTAL_SPACING;
                    y = upY;
                    z = 0;
                    upY += VERTICAL_SPACING;
                }}

                blockPositions.set(block.name, {{ x, y, z, height, block }});
            }});
        }}

        // Check if point is inside block (2D hit test)
        function isPointInBlock(px, py, corners) {{
            // Simple bounding box check using top face
            const minX = Math.min(corners.topFrontLeft.x, corners.topBackLeft.x, corners.topFrontRight.x, corners.topBackRight.x);
            const maxX = Math.max(corners.topFrontLeft.x, corners.topBackLeft.x, corners.topFrontRight.x, corners.topBackRight.x);
            const minY = Math.min(corners.topFrontLeft.y, corners.topBackLeft.y, corners.botFrontLeft.y) - 20;
            const maxY = Math.max(corners.botFrontLeft.y, corners.botBackLeft.y, corners.botFrontRight.y);

            return px >= minX && px <= maxX && py >= minY && py <= maxY;
        }}

        // Draw the entire scene
        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw title labels
            ctx.font = 'bold 16px Segoe UI';
            ctx.fillStyle = '#4a90d9';
            ctx.textAlign = 'center';
            ctx.fillText('ENCODER (Down)', ISO.offsetX - HORIZONTAL_SPACING, 30);
            ctx.fillStyle = '#d94a4a';
            ctx.fillText('BOTTLENECK', ISO.offsetX, 30);
            ctx.fillStyle = '#4ad94a';
            ctx.fillText('DECODER (Up)', ISO.offsetX + HORIZONTAL_SPACING, 30);

            // Draw flow arrows
            ctx.font = '24px Arial';
            ctx.fillStyle = '#666';
            const arrowY = 50;
            ctx.fillText('→', ISO.offsetX - HORIZONTAL_SPACING/2, arrowY);
            ctx.fillText('→', ISO.offsetX + HORIZONTAL_SPACING/2, arrowY);

            // Draw blocks (sorted for proper z-order)
            const sortedBlocks = Array.from(blockPositions.entries()).sort((a, b) => {{
                return (b[1].z + b[1].y) - (a[1].z + a[1].y);
            }});

            // First pass: draw skip connections
            connections.forEach(conn => {{
                const fromPos = blockPositions.get(conn.from);
                const toPos = blockPositions.get(conn.to);
                if (fromPos && toPos && blockCorners.has(conn.from) && blockCorners.has(conn.to)) {{
                    drawSkipConnection(blockCorners.get(conn.from), blockCorners.get(conn.to), true);
                }}
            }});

            // Second pass: draw blocks
            sortedBlocks.forEach(([name, pos]) => {{
                const block = pos.block;
                const colorScheme = colors[block.type];
                const isHovered = hoveredBlock === name;

                const corners = drawIsoBox(
                    pos.x, pos.y, pos.z,
                    BLOCK_WIDTH, BLOCK_DEPTH, pos.height,
                    colorScheme.fill, colorScheme.stroke, colorScheme.glow,
                    isHovered
                );

                blockCorners.set(name, corners);

                // Draw label
                if (labelsEnabled) {{
                    const labelPos = isoProject(pos.x + BLOCK_WIDTH/2, pos.y - pos.height - 10, pos.z + BLOCK_DEPTH/2);
                    ctx.font = isHovered ? 'bold 12px Segoe UI' : '11px Segoe UI';
                    ctx.fillStyle = isHovered ? '#fff' : '#aaa';
                    ctx.textAlign = 'center';
                    ctx.fillText(block.resolution + 'x' + block.resolution, labelPos.x, labelPos.y);
                }}
            }});

            // Draw particles
            if (animationEnabled) {{
                particles.forEach(p => {{
                    p.update();
                    p.draw();
                }});
            }}

            // Update animation
            animationPhase = (animationPhase + 1) % 100;
        }}

        // Initialize particles
        function initParticles() {{
            particles = [];

            // Create path through down blocks
            const downPath = [];
            const upPath = [];

            blocksData.forEach(block => {{
                const pos = blockPositions.get(block.name);
                if (pos) {{
                    const center = isoProject(pos.x + BLOCK_WIDTH/2, pos.y - pos.height/2, pos.z + BLOCK_DEPTH/2);
                    if (block.type === 'down') {{
                        downPath.push(center);
                    }} else if (block.type === 'up') {{
                        upPath.push(center);
                    }}
                }}
            }});

            // Add particles for down path
            for (let i = 0; i < 5; i++) {{
                particles.push(new Particle(downPath, '#4a90d9'));
            }}

            // Add particles for up path
            for (let i = 0; i < 5; i++) {{
                particles.push(new Particle(upPath.reverse(), '#4ad94a'));
            }}
        }}

        // Update info panel
        function updateInfoPanel(block) {{
            const infoDiv = document.getElementById('blockInfo');

            if (!block) {{
                infoDiv.innerHTML = '<p style="color: #666;">Hover over a block to see details</p>';
                return;
            }}

            let html = `
                <div class="title" style="color: ${{colors[block.type].fill}}">${{block.name}}</div>
                <div class="stat-row">
                    <span class="stat-label">Resolution</span>
                    <span class="stat-value">${{block.resolution}}x${{block.resolution}}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Channels</span>
                    <span class="stat-value">${{block.channels}}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Attention Layers</span>
                    <span class="stat-value">${{block.attention_count}}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Transformer Blocks</span>
                    <span class="stat-value">${{block.transformer_blocks}}</span>
                </div>
            `;

            if (block.stats && Object.keys(block.stats).length > 0) {{
                html += `
                    <h3 style="margin-top: 15px; margin-bottom: 10px; color: #888;">Activation Stats</h3>
                    <div class="stat-row">
                        <span class="stat-label">Mean</span>
                        <span class="stat-value">${{block.stats.mean?.toFixed(4) || 'N/A'}}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Std</span>
                        <span class="stat-value">${{block.stats.std?.toFixed(4) || 'N/A'}}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Range</span>
                        <span class="stat-value">${{block.stats.min?.toFixed(2) || 'N/A'}} to ${{block.stats.max?.toFixed(2) || 'N/A'}}</span>
                    </div>
                `;
            }}

            if (block.preview) {{
                html += `
                    <div class="preview-container">
                        <h3 style="margin-bottom: 10px; color: #888;">Feature Preview</h3>
                        <img src="data:image/png;base64,${{block.preview}}" alt="Feature preview">
                    </div>
                `;
            }}

            infoDiv.innerHTML = html;
        }}

        // Mouse handling
        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            let found = null;
            blockCorners.forEach((corners, name) => {{
                if (isPointInBlock(x, y, corners)) {{
                    found = name;
                }}
            }});

            if (found !== hoveredBlock) {{
                hoveredBlock = found;
                if (found) {{
                    const block = blockPositions.get(found).block;
                    updateInfoPanel(block);
                    canvas.style.cursor = 'pointer';
                }} else {{
                    updateInfoPanel(null);
                    canvas.style.cursor = 'default';
                }}
            }}
        }});

        // Control functions
        function toggleAnimation() {{
            animationEnabled = !animationEnabled;
        }}

        function toggleLabels() {{
            labelsEnabled = !labelsEnabled;
        }}

        function resetView() {{
            hoveredBlock = null;
            updateInfoPanel(null);
        }}

        // Animation loop
        function animate() {{
            draw();
            requestAnimationFrame(animate);
        }}

        // Initialize
        calculateBlockPositions();
        initParticles();
        animate();
    </script>
</body>
</html>'''

        with open(output_path, 'w') as f:
            f.write(html)

        return output_path


def create_sdxl_visualization(
    unet,
    captured_features: Optional[Dict[str, torch.Tensor]] = None,
    output_path: str = "unet_viz.html"
) -> str:
    """
    Create visualization from SDXL UNet.

    Args:
        unet: The SDXL UNet model
        captured_features: Optional dict of captured feature tensors
        output_path: Where to save the HTML

    Returns:
        Path to generated HTML file
    """
    viz = UNetVisualizer()

    # SDXL architecture info
    resolutions = [128, 128, 64, 32]  # down_blocks
    up_resolutions = [32, 64, 128, 128]  # up_blocks

    # Add down blocks
    for i, block in enumerate(unet.down_blocks):
        channels = block.resnets[0].out_channels if hasattr(block, 'resnets') else 320
        attention = len(block.attentions) if hasattr(block, 'attentions') else 0
        transformer = 0
        if attention > 0 and hasattr(block.attentions[0], 'transformer_blocks'):
            transformer = len(block.attentions[0].transformer_blocks)

        features = captured_features.get(f'down_blocks.{i}') if captured_features else None

        viz.add_block(
            name=f'down_{i}',
            block_type='down',
            resolution=resolutions[i],
            channels=channels,
            features=features,
            attention_count=attention,
            transformer_blocks=transformer
        )

    # Add mid block
    mid_channels = unet.mid_block.resnets[0].out_channels if hasattr(unet.mid_block, 'resnets') else 1280
    mid_attention = len(unet.mid_block.attentions) if hasattr(unet.mid_block, 'attentions') else 0
    mid_features = captured_features.get('mid_block') if captured_features else None

    viz.add_block(
        name='mid',
        block_type='mid',
        resolution=32,
        channels=mid_channels,
        features=mid_features,
        attention_count=mid_attention,
        transformer_blocks=10  # SDXL mid block typically has 10
    )

    # Add up blocks
    for i, block in enumerate(unet.up_blocks):
        channels = block.resnets[0].out_channels if hasattr(block, 'resnets') else 320
        attention = len(block.attentions) if hasattr(block, 'attentions') else 0
        transformer = 0
        if attention > 0 and hasattr(block.attentions[0], 'transformer_blocks'):
            transformer = len(block.attentions[0].transformer_blocks)

        features = captured_features.get(f'up_blocks.{i}') if captured_features else None

        viz.add_block(
            name=f'up_{i}',
            block_type='up',
            resolution=up_resolutions[i],
            channels=channels,
            features=features,
            attention_count=attention,
            transformer_blocks=transformer
        )

    # Add skip connections (down -> up)
    viz.add_skip_connection('down_0', 'up_2')
    viz.add_skip_connection('down_1', 'up_1')
    viz.add_skip_connection('down_2', 'up_0')

    return viz.generate_html(output_path)


# Demo function
def demo():
    """Generate demo visualization without a model."""
    viz = UNetVisualizer()

    # Add demo blocks
    demo_blocks = [
        ('down_0', 'down', 128, 320, 2, 2),
        ('down_1', 'down', 64, 640, 2, 10),
        ('down_2', 'down', 32, 1280, 2, 10),
        ('mid', 'mid', 32, 1280, 1, 10),
        ('up_0', 'up', 32, 1280, 2, 10),
        ('up_1', 'up', 64, 640, 2, 10),
        ('up_2', 'up', 128, 320, 2, 2),
    ]

    for name, btype, res, ch, att, trans in demo_blocks:
        # Create fake feature tensor for preview
        fake_features = torch.randn(1, ch, res, res)
        viz.add_block(name, btype, res, ch, fake_features, att, trans)

    # Add skip connections
    viz.add_skip_connection('down_0', 'up_2')
    viz.add_skip_connection('down_1', 'up_1')
    viz.add_skip_connection('down_2', 'up_0')

    output_path = viz.generate_html('unet_demo.html')
    print(f"Demo visualization saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    demo()
