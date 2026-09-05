"""
make_sample_video.py - Generate a synthetic test clip for benchmarking.

The benchmark needs a real video file (ingestion uses cv2.VideoCapture), but we
don't want an external download. This writes a deterministic synthetic clip
(drifting color gradient + moving circles + frame counter) at a chosen
resolution and length.

Usage:
    python make_sample_video.py                         # 3000 frames @ 1280x720
    python make_sample_video.py --num-frames 9000 --width 1920 --height 1080
"""

import argparse

import cv2
import numpy as np


def make_sample_video(
    path: str = "sample_video.mp4",
    width: int = 1280,
    height: int = 720,
    num_frames: int = 3000,
    fps: int = 30,
) -> str:
    """Write a synthetic .mp4 and return its path."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")

    ramp = np.arange(width, dtype=np.uint16)
    for i in range(num_frames):
        # drifting horizontal gradient, colorized
        row = ((ramp + i * 3) % 256).astype(np.uint8)
        gray = np.tile(row, (height, 1))
        frame = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        # a handful of moving circles so there is real high-frequency detail
        for k in range(5):
            cx = int((i * 7 + k * 251) % width)
            cy = int(height / 2 + (height / 3) * np.sin((i + k * 40) / 25.0))
            color = (int(50 * k) % 255, (255 - 40 * k) % 255, (90 * k) % 255)
            cv2.circle(frame, (cx, cy), 40, color, -1)

        cv2.putText(
            frame, f"frame {i}", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    print(f"✓ wrote {path}  ({width}x{height}, {num_frames} frames, {fps} fps)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic benchmark clip")
    parser.add_argument("--output", default="sample_video.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--num-frames", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    make_sample_video(
        args.output, args.width, args.height, args.num_frames, args.fps
    )


if __name__ == "__main__":
    main()
