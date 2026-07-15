"""Read/write helpers for the injection engine: video frames, audio arrays, and A/V muxing."""

import cv2
import ffmpeg
import numpy as np
import soundfile as sf


def read_video_frames(path: str) -> tuple:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def write_video_frames(frames: list, fps: float, path: str) -> None:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def read_audio(path: str) -> tuple:
    samples, sr = sf.read(path, always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples.astype(np.float32), sr


def write_audio(samples: np.ndarray, sr: int, path: str) -> None:
    sf.write(path, samples, sr)


def mux_video_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Combine a video-only file and an audio file into one container.

    Video is stream-copied (untouched); audio is re-encoded to AAC since raw PCM
    doesn't fit in an MP4 container. `shortest` trims the output to the shorter
    of the two input durations so a slightly longer audio track (from padding
    during desync injection) doesn't leave a trailing silent/frozen tail.
    """
    video_in = ffmpeg.input(video_path)
    audio_in = ffmpeg.input(audio_path)
    (
        ffmpeg.output(video_in["v"], audio_in["a"], output_path, vcodec="copy", acodec="aac", shortest=None)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
