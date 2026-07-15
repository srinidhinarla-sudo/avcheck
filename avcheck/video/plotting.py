"""Quality-curve plotting for video scoring results."""

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, no display needed for CI/CLI use
import matplotlib.pyplot as plt
import numpy as np


def plot_quality_curve(per_frame: list, output_path: str) -> None:
    """Plot PSNR and SSIM over time as a two-panel figure and save to output_path."""
    timestamps = [f["timestamp"] for f in per_frame]
    psnr_values = [f["psnr"] if np.isfinite(f["psnr"]) else np.nan for f in per_frame]
    ssim_values = [f["ssim"] for f in per_frame]

    fig, (ax_psnr, ax_ssim) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_psnr.plot(timestamps, psnr_values, color="tab:blue")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_title("AVCheck Video Quality Curve")
    ax_psnr.grid(alpha=0.3)

    ax_ssim.plot(timestamps, ssim_values, color="tab:orange")
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.set_xlabel("Time (s)")
    ax_ssim.set_ylim(0, 1.05)
    ax_ssim.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
