"""Tee-style ffmpeg recorder for closed_loop_cap rollouts.

Goals:
    1. Record the whole episode to a single mp4 (episode.mp4).
    2. Record each subtask's window to a separate mp4 (subtask_XX/rollout.mp4).
    3. Reuse RoboTwin's Base_Task.take_action hook that writes RGB bytes to
       `self.eval_video_ffmpeg.stdin`. We substitute a proxy object whose
       .stdin.write() forwards ("tees") the bytes to every currently-active
       ffmpeg process.
    4. Graceful degradation when ffmpeg is unavailable — recorder reports
       `available=False` and the orchestrator can skip video writes instead
       of crashing.

Non-goals:
    - Synchronizing wall-clock fps with simulator steps (we rely on the
       fixed framerate encoded at ffmpeg's -framerate flag; each take_action
       writes exactly one frame).
    - Audio.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# -----------------------------
# ffmpeg discovery
# -----------------------------


def find_ffmpeg_binary() -> str | None:
    """Locate an ffmpeg binary, preferring the imageio-ffmpeg bundled copy.

    The bundled copy lets us run inside conda envs that don't have a system
    ffmpeg without any manual install step.
    """
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001
        pass
    path = shutil.which("ffmpeg")
    return path if path else None


# -----------------------------
# Camera size discovery
# -----------------------------


_BUILTIN_CAMERA_SIZES = {
    "D435": (320, 240),
    "Large_D435": (640, 480),
    "L515": (320, 180),
    "Large_L515": (640, 360),
}


def camera_size_from_config(config: dict) -> tuple[int, int]:
    """Return (width, height) for the configured head-camera type."""
    cam_type = (
        config.get("env", {})
        .get("camera", {})
        .get("head_camera_type", "D435")
    )
    size = _BUILTIN_CAMERA_SIZES.get(cam_type)
    if size is None:
        logger.warning("unknown camera type %r; defaulting to D435 320x240", cam_type)
        return _BUILTIN_CAMERA_SIZES["D435"]
    return size


# -----------------------------
# stdin tee
# -----------------------------


class _StdinTee:
    """Write-through proxy for multiple stdin pipes.

    Intentionally silent on per-pipe BrokenPipeError so one failed ffmpeg
    doesn't kill the rest of the recording. Failed pipes are removed on the
    spot.
    """

    def __init__(self) -> None:
        self._pipes: list[Any] = []

    def add(self, pipe: Any) -> None:
        self._pipes.append(pipe)

    def remove(self, pipe: Any) -> None:
        try:
            self._pipes.remove(pipe)
        except ValueError:
            pass

    def write(self, data: bytes) -> int:
        dead: list[Any] = []
        for p in self._pipes:
            try:
                p.write(data)
            except (BrokenPipeError, ValueError, OSError) as exc:
                logger.warning("dropping ffmpeg pipe after write failure: %s", exc)
                dead.append(p)
        for p in dead:
            self.remove(p)
        return len(data)

    def flush(self) -> None:
        for p in self._pipes:
            try:
                p.flush()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        for p in self._pipes:
            try:
                p.close()
            except Exception:  # noqa: BLE001
                pass
        self._pipes.clear()


# -----------------------------
# Recorder
# -----------------------------


@dataclass
class _ChannelHandle:
    proc: subprocess.Popen
    path: Path


class FfmpegRecorder:
    """Duck-typed substitute for `subprocess.Popen` that Base_Task's
    `_set_eval_video_ffmpeg` expects.

    Base_Task contract (envs/_base_task.py):
        - `self.eval_video_ffmpeg.stdin.write(bytes)`  (take_action, line 1485)
        - `self.eval_video_ffmpeg.stdin.close()`       (_del_eval_video_ffmpeg)
        - `self.eval_video_ffmpeg.wait()`              (_del_eval_video_ffmpeg)

    We satisfy this by exposing:
        - `.stdin` — a `_StdinTee` that fans out writes to every active ffmpeg.
        - `.wait()` — flushes+closes every active ffmpeg with a bounded wait.
    """

    def __init__(
        self,
        *,
        video_size: tuple[int, int],
        framerate: int = 10,
        crf: int = 23,
        ffmpeg_binary: str | None = None,
    ) -> None:
        self.video_size = video_size
        self.framerate = int(framerate)
        self.crf = int(crf)
        self.ffmpeg_binary = ffmpeg_binary or find_ffmpeg_binary()
        self.available = self.ffmpeg_binary is not None
        if not self.available:
            logger.warning(
                "no ffmpeg binary found — video recording disabled "
                "(pip install imageio-ffmpeg to enable)"
            )
        self.stdin = _StdinTee()
        self._episode: _ChannelHandle | None = None
        self._subtask: _ChannelHandle | None = None

    # -- internal --

    def _build_cmd(self, out_path: Path) -> list[str]:
        w, h = self.video_size
        return [
            str(self.ffmpeg_binary),
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{w}x{h}",
            "-framerate", str(self.framerate),
            "-i", "-",
            "-pix_fmt", "yuv420p",
            "-vcodec", "libx264",
            "-crf", str(self.crf),
            str(out_path),
        ]

    def _spawn(self, out_path: Path) -> _ChannelHandle | None:
        if not self.available:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.Popen(  # noqa: S603 — bin path is our own discovery
                self._build_cmd(out_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError) as exc:
            logger.error("failed to spawn ffmpeg at %s: %s", self.ffmpeg_binary, exc)
            self.available = False
            return None
        assert proc.stdin is not None
        self.stdin.add(proc.stdin)
        return _ChannelHandle(proc=proc, path=out_path)

    @staticmethod
    def _close_channel(ch: _ChannelHandle, stdin_tee: _StdinTee, timeout: float = 5.0) -> None:
        try:
            stdin_tee.remove(ch.proc.stdin)
        except Exception:  # noqa: BLE001
            pass
        try:
            if ch.proc.stdin is not None and not ch.proc.stdin.closed:
                ch.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            ch.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg did not exit in %.1fs for %s, killing", timeout, ch.path)
            ch.proc.kill()
            try:
                ch.proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass

    # -- episode-level --

    def start_episode(self, mp4_path: Path) -> Path | None:
        if self._episode is not None:
            logger.warning("start_episode called but a recording is already active")
            return self._episode.path
        ch = self._spawn(Path(mp4_path))
        self._episode = ch
        return ch.path if ch else None

    def end_episode(self) -> Path | None:
        if self._episode is None:
            return None
        path = self._episode.path
        self._close_channel(self._episode, self.stdin)
        self._episode = None
        return path

    # -- subtask-level --

    def start_subtask(self, mp4_path: Path) -> Path | None:
        """Begin a new subtask segment. Closes any prior segment first."""
        if self._subtask is not None:
            self.end_subtask()
        ch = self._spawn(Path(mp4_path))
        self._subtask = ch
        return ch.path if ch else None

    def end_subtask(self) -> Path | None:
        if self._subtask is None:
            return None
        path = self._subtask.path
        self._close_channel(self._subtask, self.stdin)
        self._subtask = None
        return path

    # -- ffmpeg-like proxy for Base_Task --

    def wait(self, timeout: float | None = None) -> int:
        """Called by Base_Task._del_eval_video_ffmpeg. Close everything."""
        # At this point Base_Task has already closed .stdin — but we treat that
        # as "close every channel we still own" so orphan processes don't linger.
        to = float(timeout) if timeout is not None else 5.0
        if self._subtask is not None:
            self._close_channel(self._subtask, self.stdin, timeout=to)
            self._subtask = None
        if self._episode is not None:
            self._close_channel(self._episode, self.stdin, timeout=to)
            self._episode = None
        return 0

    def __bool__(self) -> bool:
        # Base_Task uses `if self.eval_video_ffmpeg:` — a recorder with no
        # active channels still returns True so Base_Task's cleanup runs.
        return True
