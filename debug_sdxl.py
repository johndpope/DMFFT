"""
SDXL Debug Visualization Module
===============================

Provides comprehensive debugging and visualization tools for understanding
SDXL U-Net internals during image generation.

Features:
- Feature map visualization (hidden_states, skip connections)
- Frequency domain analysis (FFT magnitude/phase)
- Latent evolution tracking across denoising steps
- Attention map visualization
- Block-by-block feature dumps
"""

import os
import torch
import torch.fft as fft
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from logger import logger

# Enable lovely_tensors for colorful tensor visualization
try:
    import lovely_tensors as lt
    lt.monkey_patch()
    LOVELY_TENSORS_AVAILABLE = True
except ImportError:
    LOVELY_TENSORS_AVAILABLE = False
    logger.warning("lovely_tensors not available, using standard visualization")


def _get_set_debug_capture():
    """Lazy import to avoid circular dependency."""
    try:
        from utils_sdxl import set_debug_capture
        return set_debug_capture
    except ImportError:
        return None


@dataclass
class DebugConfig:
    """Configuration for debug visualization."""
    enabled: bool = False
    output_dir: str = "debug_output"
    save_features: bool = True          # Save feature maps as images
    save_fft: bool = True               # Save FFT magnitude spectra
    save_latents: bool = True           # Save latent evolution
    save_skip_connections: bool = True  # Save skip connection features
    sample_channels: int = 16           # Number of channels to visualize
    timestep_interval: int = 5          # Save every N timesteps
    normalize_per_channel: bool = True  # Normalize each channel independently


class SDXLDebugger:
    """
    Debug visualization for SDXL U-Net internals.

    Captures and visualizes:
    - Down block features (encoder path)
    - Mid block features (bottleneck)
    - Up block features (decoder path)
    - Skip connections
    - Latent evolution during denoising
    """

    def __init__(self, config: Optional[DebugConfig] = None):
        self.config = config or DebugConfig()
        self.captured_features: Dict[str, List[torch.Tensor]] = {}
        self.latent_history: List[torch.Tensor] = []
        self.current_timestep: int = 0
        self.generation_id: int = 0
        self._hooks: List = []

        if self.config.enabled:
            os.makedirs(self.config.output_dir, exist_ok=True)
            self._enable_feature_capture()

    def _enable_feature_capture(self):
        """Enable feature capture in utils_sdxl."""
        set_debug_capture = _get_set_debug_capture()
        if set_debug_capture:
            set_debug_capture(True, self._feature_capture_callback)
            logger.debug("Feature capture enabled in utils_sdxl")

    def _disable_feature_capture(self):
        """Disable feature capture in utils_sdxl."""
        set_debug_capture = _get_set_debug_capture()
        if set_debug_capture:
            set_debug_capture(False, None)

    def _feature_capture_callback(
        self,
        name: str,
        hidden_states: torch.Tensor,
        res_hidden_states: Optional[torch.Tensor],
        block_idx: int
    ):
        """Callback invoked by utils_sdxl to capture features."""
        if not self.config.enabled or not self.config.save_features:
            return

        # Only capture at configured timestep intervals
        if self.current_timestep % self.config.timestep_interval != 0:
            return

        self.capture_features(name, hidden_states, res_hidden_states, block_idx)

    def reset(self):
        """Reset captured data for new generation."""
        self.captured_features.clear()
        self.latent_history.clear()
        self.current_timestep = 0
        self.generation_id += 1

        if self.config.enabled:
            gen_dir = Path(self.config.output_dir) / f"gen_{self.generation_id:04d}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            self._enable_feature_capture()

    @property
    def gen_dir(self) -> Path:
        return Path(self.config.output_dir) / f"gen_{self.generation_id:04d}"

    def capture_latent(self, latent: torch.Tensor, timestep: int):
        """Capture latent at current denoising step."""
        if not self.config.enabled or not self.config.save_latents:
            return

        self.current_timestep = timestep

        if timestep % self.config.timestep_interval == 0:
            self.latent_history.append(latent.detach().cpu().clone())
            self._save_latent_visualization(latent, timestep)

    def capture_features(
        self,
        name: str,
        hidden_states: torch.Tensor,
        res_hidden_states: Optional[torch.Tensor] = None,
        block_idx: int = 0
    ):
        """
        Capture feature maps from U-Net blocks.

        Args:
            name: Block identifier (e.g., "up_block_0", "down_block_1")
            hidden_states: Main backbone features [B, C, H, W]
            res_hidden_states: Skip connection features [B, C, H, W]
            block_idx: ResNet block index within the up/down block
        """
        if not self.config.enabled or not self.config.save_features:
            return

        key = f"{name}_block{block_idx}_t{self.current_timestep}"

        # Save backbone features
        self._save_feature_grid(
            hidden_states,
            self.gen_dir / f"{key}_backbone.png",
            title=f"{name} Backbone (t={self.current_timestep})"
        )

        # Save skip connection features
        if res_hidden_states is not None and self.config.save_skip_connections:
            self._save_feature_grid(
                res_hidden_states,
                self.gen_dir / f"{key}_skip.png",
                title=f"{name} Skip (t={self.current_timestep})"
            )

        # Save FFT analysis
        if self.config.save_fft:
            self._save_fft_analysis(
                hidden_states,
                self.gen_dir / f"{key}_backbone_fft.png",
                title=f"{name} Backbone FFT"
            )
            if res_hidden_states is not None:
                self._save_fft_analysis(
                    res_hidden_states,
                    self.gen_dir / f"{key}_skip_fft.png",
                    title=f"{name} Skip FFT"
                )

    def _normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] range."""
        x = x.float()
        if self.config.normalize_per_channel:
            # Normalize each channel independently
            b, c, h, w = x.shape
            x_flat = x.view(b, c, -1)
            x_min = x_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
            x_max = x_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
            x = (x - x_min) / (x_max - x_min + 1e-8)
        else:
            x_min, x_max = x.min(), x.max()
            x = (x - x_min) / (x_max - x_min + 1e-8)
        return x

    def _tensor_to_grid(
        self,
        x: torch.Tensor,
        nrow: int = 4
    ) -> np.ndarray:
        """
        Convert tensor channels to image grid.

        Args:
            x: Tensor [B, C, H, W] - takes first batch
            nrow: Number of images per row

        Returns:
            Grid image as numpy array [H, W, 3]
        """
        x = x[0].detach().cpu()  # Take first batch
        c, h, w = x.shape

        # Sample channels if too many
        n_channels = min(c, self.config.sample_channels)
        indices = torch.linspace(0, c - 1, n_channels).long()
        x = x[indices]

        # Normalize
        x = self._normalize_tensor(x.unsqueeze(0)).squeeze(0)

        # Create grid
        ncol = nrow
        nrow_actual = (n_channels + ncol - 1) // ncol

        grid = np.zeros((nrow_actual * h, ncol * w), dtype=np.float32)

        # Use colormap for vibrant visualization
        cmap = plt.get_cmap('viridis')

        for idx in range(n_channels):
            row = idx // ncol
            col = idx % ncol
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = x[idx].numpy()

        # Apply colormap for colorful output
        grid_rgb = cmap(grid)[:, :, :3]  # Get RGB, discard alpha
        return (grid_rgb * 255).astype(np.uint8)

    def _save_feature_grid(
        self,
        x: torch.Tensor,
        path: Path,
        title: str = ""
    ):
        """Save feature maps as image grid using lovely_tensors for color."""
        try:
            b, c, h, w = x.shape
            logger.debug(f"Saving {title}: shape={x.shape}, path={path}")

            if LOVELY_TENSORS_AVAILABLE:
                # Use lovely_tensors for colorful visualization
                self._save_with_lovely_tensors(x, path, title)
            else:
                # Fallback to standard colormap
                grid = self._tensor_to_grid(x)
                img = Image.fromarray(grid)
                img.save(path)
        except Exception as e:
            logger.error(f"Failed to save feature grid: {e}")

    def _save_with_lovely_tensors(
        self,
        x: torch.Tensor,
        path: Path,
        title: str = ""
    ):
        """Save tensor visualization using lovely_tensors."""
        import lovely_tensors as lt

        x = x[0].detach().cpu()  # Take first batch [C, H, W]
        c, h, w = x.shape

        # Sample channels
        n_channels = min(c, self.config.sample_channels)
        indices = torch.linspace(0, c - 1, n_channels).long()
        x_sampled = x[indices]

        # Create figure with subplots
        ncol = 4
        nrow = (n_channels + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
        if nrow == 1:
            axes = axes.reshape(1, -1)

        # Use different colormaps for visual variety
        colormaps = ['viridis', 'plasma', 'magma', 'inferno', 'cividis', 'turbo']

        for idx in range(n_channels):
            row, col = idx // ncol, idx % ncol
            ax = axes[row, col]

            # Normalize channel
            ch = x_sampled[idx]
            ch_norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

            # Apply colormap - cycle through colormaps
            cmap = plt.get_cmap(colormaps[idx % len(colormaps)])
            im = ax.imshow(ch_norm.numpy(), cmap=cmap, aspect='auto')
            ax.set_title(f'Ch {indices[idx].item()}', fontsize=8, color='white')
            ax.axis('off')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(n_channels, nrow * ncol):
            row, col = idx // ncol, idx % ncol
            axes[row, col].axis('off')

        fig.suptitle(title, fontsize=10, color='white')
        fig.patch.set_facecolor('#1a1a2e')

        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor('#1a1a2e')

        plt.tight_layout()
        plt.savefig(path, facecolor='#1a1a2e', edgecolor='none', dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _save_latent_visualization(self, latent: torch.Tensor, timestep: int):
        """Save latent visualization with colorful colormaps and colorbars."""
        try:
            path = self.gen_dir / f"latent_t{timestep:04d}.png"

            # Latent is [B, 4, H, W] - visualize all 4 channels
            lat = latent[0].detach().cpu()

            # Create figure with 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            colormaps = ['plasma', 'viridis', 'magma', 'inferno']
            channel_names = ['Latent Ch 0', 'Latent Ch 1', 'Latent Ch 2', 'Latent Ch 3']

            for i in range(4):
                row, col = i // 2, i % 2
                ax = axes[row, col]

                ch = lat[i]
                ch_norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

                cmap = plt.get_cmap(colormaps[i])
                im = ax.imshow(ch_norm.numpy(), cmap=cmap, aspect='auto')
                ax.set_title(f'{channel_names[i]} (t={timestep})', fontsize=10, color='white')
                ax.axis('off')

                # Add colorbar with stats
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(colors='white', labelsize=8)

            fig.suptitle(f'Latent Space - Timestep {timestep}', fontsize=12, color='cyan')
            fig.patch.set_facecolor('#1a1a2e')

            for ax_row in axes:
                for ax in ax_row:
                    ax.set_facecolor('#1a1a2e')

            plt.tight_layout()
            plt.savefig(path, facecolor='#1a1a2e', edgecolor='none', dpi=100, bbox_inches='tight')
            plt.close(fig)

            logger.debug(f"Saved latent t={timestep} to {path}")
        except Exception as e:
            logger.error(f"Failed to save latent: {e}")

    def _save_fft_analysis(
        self,
        x: torch.Tensor,
        path: Path,
        title: str = ""
    ):
        """Save FFT magnitude spectrum visualization with colorful colormaps."""
        try:
            x = x[0].detach().cpu().float()  # First batch
            c, h, w = x.shape

            # Sample channels
            n_channels = min(c, self.config.sample_channels)
            indices = torch.linspace(0, c - 1, n_channels).long()
            x_sampled = x[indices]

            # Create figure with subplots
            ncol = 4
            nrow = (n_channels + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
            if nrow == 1:
                axes = axes.reshape(1, -1)

            # Spectral colormaps for FFT visualization
            fft_colormaps = ['turbo', 'jet', 'nipy_spectral', 'gist_rainbow', 'rainbow', 'hsv']

            for idx in range(n_channels):
                row, col = idx // ncol, idx % ncol
                ax = axes[row, col]

                ch = x_sampled[idx]
                ch_fft = fft.fft2(ch)
                ch_fft = fft.fftshift(ch_fft)
                mag = torch.abs(ch_fft)
                # Log scale for better visualization
                mag = torch.log1p(mag)
                # Normalize
                mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

                # Apply colormap - cycle through spectral colormaps
                cmap = plt.get_cmap(fft_colormaps[idx % len(fft_colormaps)])
                im = ax.imshow(mag.numpy(), cmap=cmap, aspect='auto')
                ax.set_title(f'Ch {indices[idx].item()} FFT', fontsize=8, color='white')
                ax.axis('off')

                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Hide unused subplots
            for idx in range(n_channels, nrow * ncol):
                row, col = idx // ncol, idx % ncol
                axes[row, col].axis('off')

            fig.suptitle(f'{title} - Frequency Domain', fontsize=10, color='white')
            fig.patch.set_facecolor('#0a0a1a')

            for ax_row in axes:
                for ax in ax_row:
                    ax.set_facecolor('#0a0a1a')

            plt.tight_layout()
            plt.savefig(path, facecolor='#0a0a1a', edgecolor='none', dpi=100, bbox_inches='tight')
            plt.close(fig)

            logger.debug(f"Saved FFT analysis to {path}")
        except Exception as e:
            logger.error(f"Failed to save FFT analysis: {e}")

    def create_latent_evolution_gif(self):
        """Create animated GIF showing latent evolution with colorful matplotlib frames."""
        if not self.config.enabled or not self.latent_history:
            return

        try:
            import io
            frames = []
            colormaps = ['plasma', 'viridis', 'magma', 'inferno']
            channel_names = ['Ch 0', 'Ch 1', 'Ch 2', 'Ch 3']

            total_steps = len(self.latent_history)

            for i, lat in enumerate(self.latent_history):
                lat = lat[0]  # First batch

                # Create figure with 2x2 subplots
                fig, axes = plt.subplots(2, 2, figsize=(8, 8))

                for j in range(4):
                    row, col = j // 2, j % 2
                    ax = axes[row, col]

                    ch = lat[j]
                    ch_norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

                    cmap = plt.get_cmap(colormaps[j])
                    im = ax.imshow(ch_norm.numpy(), cmap=cmap, aspect='auto')
                    ax.set_title(channel_names[j], fontsize=10, color='white')
                    ax.axis('off')

                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(colors='white', labelsize=6)

                # Calculate progress and timestep info
                progress = (i + 1) / total_steps * 100
                fig.suptitle(f'Latent Evolution - Step {i+1}/{total_steps} ({progress:.0f}%)',
                           fontsize=12, color='cyan', fontweight='bold')

                fig.patch.set_facecolor('#1a1a2e')
                for ax_row in axes:
                    for ax in ax_row:
                        ax.set_facecolor('#1a1a2e')

                plt.tight_layout()

                # Save figure to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', facecolor='#1a1a2e',
                           edgecolor='none', dpi=80, bbox_inches='tight')
                buf.seek(0)
                frames.append(Image.open(buf).copy())
                buf.close()
                plt.close(fig)

            if frames:
                gif_path = self.gen_dir / "latent_evolution.gif"
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=300,  # 300ms per frame
                    loop=0
                )
                logger.info(f"Saved latent evolution GIF to {gif_path}")
        except Exception as e:
            logger.error(f"Failed to create latent evolution GIF: {e}")

    def summarize_generation(self) -> Dict:
        """Generate summary of captured debug data."""
        summary = {
            "generation_id": self.generation_id,
            "timesteps_captured": len(self.latent_history),
            "features_captured": list(self.captured_features.keys()),
            "output_dir": str(self.gen_dir) if self.config.enabled else None
        }

        if self.config.enabled:
            logger.info(f"Debug summary: {summary}")

        return summary


# Global debugger instance
_debugger: Optional[SDXLDebugger] = None


def get_debugger() -> SDXLDebugger:
    """Get or create global debugger instance."""
    global _debugger
    if _debugger is None:
        _debugger = SDXLDebugger()
    return _debugger


def enable_debug(
    output_dir: str = "debug_output",
    save_features: bool = True,
    save_fft: bool = True,
    save_latents: bool = True,
    timestep_interval: int = 5,
    sample_channels: int = 16
):
    """Enable debug visualization with specified settings."""
    global _debugger
    config = DebugConfig(
        enabled=True,
        output_dir=output_dir,
        save_features=save_features,
        save_fft=save_fft,
        save_latents=save_latents,
        timestep_interval=timestep_interval,
        sample_channels=sample_channels
    )
    _debugger = SDXLDebugger(config)
    logger.info(f"Debug enabled. Output: {output_dir}")
    return _debugger


def disable_debug():
    """Disable debug visualization."""
    global _debugger
    if _debugger:
        _debugger.config.enabled = False
    logger.info("Debug disabled")


def visualize_tensor_stats(x: torch.Tensor, name: str = "tensor"):
    """Log tensor statistics for debugging."""
    with torch.no_grad():
        stats = {
            "name": name,
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "min": x.min().item(),
            "max": x.max().item(),
            "mean": x.mean().item(),
            "std": x.std().item(),
            "has_nan": torch.isnan(x).any().item(),
            "has_inf": torch.isinf(x).any().item(),
        }
        logger.debug(f"Tensor stats [{name}]: {stats}")
        return stats


def visualize_frequency_components(
    x: torch.Tensor,
    threshold: int = 2,
    save_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose tensor into low and high frequency components.

    Returns:
        Tuple of (low_freq, high_freq) tensors
    """
    x = x.float()
    x_fft = fft.fft2(x)
    x_fft = fft.fftshift(x_fft)

    b, c, h, w = x.shape

    # Create low-frequency mask
    crow, ccol = h // 2, w // 2
    lf_mask = torch.zeros((b, c, h, w), device=x.device)
    lf_mask[..., crow-threshold:crow+threshold, ccol-threshold:ccol+threshold] = 1.0

    # Separate components
    lf_fft = x_fft * lf_mask
    hf_fft = x_fft * (1 - lf_mask)

    # Transform back
    lf = fft.ifft2(fft.ifftshift(lf_fft)).real
    hf = fft.ifft2(fft.ifftshift(hf_fft)).real

    if save_path:
        debugger = get_debugger()
        if debugger.config.enabled:
            # Save visualization
            path = Path(save_path)
            debugger._save_feature_grid(lf, path.with_suffix(".lf.png"))
            debugger._save_feature_grid(hf, path.with_suffix(".hf.png"))

    return lf.to(x.dtype), hf.to(x.dtype)


def create_debug_callback(debugger: Optional[SDXLDebugger] = None):
    """
    Create a callback function for diffusers pipeline.

    Use with:
        pipe(..., callback=create_debug_callback(), callback_steps=1)
    """
    dbg = debugger or get_debugger()

    def callback(pipe, step: int, timestep: int, callback_kwargs: Dict):
        """Callback invoked at each denoising step."""
        if not dbg.config.enabled:
            return callback_kwargs

        latents = callback_kwargs.get("latents")
        if latents is not None:
            dbg.capture_latent(latents, timestep)

        return callback_kwargs

    return callback
