from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..models import SampleFrame


class VideoSampler:
    def __init__(self, ffmpeg_binary: str = "ffmpeg") -> None:
        self.ffmpeg_binary = ffmpeg_binary
        if shutil.which(ffmpeg_binary) is None:
            raise RuntimeError("ffmpeg is required but was not found on PATH")

    def sample(
        self,
        video_path: Path,
        video_id: str,
        center_time: float,
        before: float,
        after: float,
        fps: float,
        max_frames: int,
        output_dir: Path,
    ) -> list[SampleFrame]:
        start = max(0.0, float(center_time) - float(before))
        duration = max(0.1, float(before) + float(after))
        fps = max(0.1, float(fps))
        if duration * fps > max_frames:
            fps = max_frames / duration

        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = output_dir / "sample_%04d.jpg"
        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.6f}",
            "-vf",
            f"fps={fps:.8f}",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "2",
            str(pattern),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg sampling failed: {result.stderr.strip()}")

        paths = sorted(output_dir.glob("sample_*.jpg"))
        samples: list[SampleFrame] = []
        for index, path in enumerate(paths):
            pts_time = start + index / fps
            samples.append(
                SampleFrame(
                    sample_id=f"T{index + 1:03d}",
                    video_id=video_id,
                    pts_time=pts_time,
                    image_path=path,
                )
            )
        return samples

