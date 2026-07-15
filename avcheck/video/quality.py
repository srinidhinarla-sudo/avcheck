"""Per-frame video quality scoring: PSNR and SSIM between a reference and processed video."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_psnr(ref_frame: np.ndarray, test_frame: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB: how large the pixel-error energy is relative to
    the maximum possible pixel value, on a log scale. Higher is better; identical frames
    give +inf. It penalizes large, sparse errors and small, widespread errors similarly
    (it only sees squared error, not where or how structured it is)."""
    mse = np.mean((ref_frame.astype(np.float64) - test_frame.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return float(20 * np.log10(max_pixel / np.sqrt(mse)))


def compute_ssim(ref_frame: np.ndarray, test_frame: np.ndarray) -> float:
    """Structural Similarity Index: compares local luminance, contrast, and structure
    between corresponding image patches rather than raw pixel error. Complements PSNR
    because it catches structural/perceptual damage (blocking, blur, banding) that can
    have low pixel-error energy but still look visibly wrong to a human."""
    score, _ = ssim(ref_frame, test_frame, full=True, data_range=255)
    return float(score)


def score_video(ref_path: str, test_path: str) -> dict:
    """Walk both videos frame-by-frame (grayscale) and score PSNR/SSIM per matched pair.

    Stops at the shorter of the two videos' frame counts. Frames are compared in
    lockstep by index, not by timestamp — this assumes the two videos are frame-aligned
    (no A/V-desync-style shift), which is a documented limitation.
    """
    cap_ref = cv2.VideoCapture(ref_path)
    cap_test = cv2.VideoCapture(test_path)
    if not cap_ref.isOpened():
        raise IOError(f"Could not open reference video: {ref_path}")
    if not cap_test.isOpened():
        raise IOError(f"Could not open test video: {test_path}")

    fps = cap_ref.get(cv2.CAP_PROP_FPS) or 30.0

    per_frame = []
    frame_idx = 0
    try:
        while True:
            ret_ref, frame_ref = cap_ref.read()
            ret_test, frame_test = cap_test.read()
            if not ret_ref or not ret_test:
                break

            gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
            gray_test = cv2.cvtColor(frame_test, cv2.COLOR_BGR2GRAY)

            per_frame.append(
                {
                    "frame": frame_idx,
                    "timestamp": frame_idx / fps,
                    "psnr": compute_psnr(gray_ref, gray_test),
                    "ssim": compute_ssim(gray_ref, gray_test),
                }
            )
            frame_idx += 1
    finally:
        cap_ref.release()
        cap_test.release()

    return {"fps": fps, "per_frame": per_frame, "summary": _summarize(per_frame)}


def _summarize(per_frame: list) -> dict:
    if not per_frame:
        return {
            "num_frames": 0,
            "mean_psnr": float("inf"),
            "min_psnr": float("inf"),
            "mean_ssim": 0.0,
            "min_ssim": 0.0,
            "worst_frame_index": None,
            "worst_frame_timestamp": None,
        }

    psnr_values = np.array([f["psnr"] for f in per_frame])
    ssim_values = np.array([f["ssim"] for f in per_frame])
    finite_psnr = psnr_values[np.isfinite(psnr_values)]
    worst_idx = int(np.argmin(ssim_values))

    return {
        "num_frames": len(per_frame),
        "mean_psnr": float(np.mean(finite_psnr)) if len(finite_psnr) else float("inf"),
        "min_psnr": float(np.min(finite_psnr)) if len(finite_psnr) else float("inf"),
        "mean_ssim": float(np.mean(ssim_values)),
        "min_ssim": float(np.min(ssim_values)),
        "worst_frame_index": worst_idx,
        "worst_frame_timestamp": per_frame[worst_idx]["timestamp"],
    }
