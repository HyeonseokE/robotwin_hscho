"""Unit tests for FfmpegRecorder (no SAPIEN required).

Covers:
    - ffmpeg command construction (resolution, framerate, crf, output path)
    - camera_size_from_config fallback
    - Tee stdin forwards writes, drops broken pipes
    - Missing ffmpeg binary → available=False, no crash
    - wait() closes all channels
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.video.recorder import (  # noqa: E402
    FfmpegRecorder,
    _StdinTee,
    camera_size_from_config,
    find_ffmpeg_binary,
)


# -------------------- discovery --------------------


@pytest.mark.unit
def test_find_ffmpeg_returns_string_or_none() -> None:
    result = find_ffmpeg_binary()
    assert result is None or isinstance(result, str)


@pytest.mark.unit
def test_camera_size_defaults_to_d435() -> None:
    assert camera_size_from_config({}) == (320, 240)
    assert camera_size_from_config({"env": {}}) == (320, 240)


@pytest.mark.unit
def test_camera_size_picks_large() -> None:
    cfg = {"env": {"camera": {"head_camera_type": "Large_D435"}}}
    assert camera_size_from_config(cfg) == (640, 480)


@pytest.mark.unit
def test_camera_size_unknown_falls_back() -> None:
    cfg = {"env": {"camera": {"head_camera_type": "Imaginary_Camera"}}}
    assert camera_size_from_config(cfg) == (320, 240)


# -------------------- _StdinTee --------------------


@pytest.mark.unit
def test_tee_broadcasts_writes() -> None:
    tee = _StdinTee()
    a, b = io.BytesIO(), io.BytesIO()
    tee.add(a)
    tee.add(b)
    tee.write(b"hello")
    assert a.getvalue() == b"hello"
    assert b.getvalue() == b"hello"


@pytest.mark.unit
def test_tee_drops_broken_pipe() -> None:
    class _BrokenPipe:
        def write(self, _):
            raise BrokenPipeError("pipe closed")

    tee = _StdinTee()
    broken = _BrokenPipe()
    good = io.BytesIO()
    tee.add(broken)
    tee.add(good)
    tee.write(b"x")
    # Broken pipe is removed; subsequent write only hits the good sink.
    tee.write(b"y")
    assert good.getvalue() == b"xy"


@pytest.mark.unit
def test_tee_close_empties_pipes() -> None:
    tee = _StdinTee()
    tee.add(io.BytesIO())
    tee.add(io.BytesIO())
    tee.close()
    tee.write(b"z")   # should silently do nothing


# -------------------- FfmpegRecorder (bin missing) --------------------


@pytest.mark.unit
def test_recorder_unavailable_when_no_binary() -> None:
    rec = FfmpegRecorder(video_size=(320, 240), ffmpeg_binary=None)
    # find_ffmpeg_binary may still return the imageio-ffmpeg bundled copy;
    # force the unavailable path explicitly.
    rec.ffmpeg_binary = None
    rec.available = False
    assert rec.start_episode(Path("/tmp/nonexistent.mp4")) is None
    assert rec.start_subtask(Path("/tmp/nonexistent2.mp4")) is None
    # wait() on an empty recorder is a no-op
    assert rec.wait() == 0


# -------------------- FfmpegRecorder (mocked subprocess) --------------------


def _mock_popen(output_holder: dict):
    def factory(cmd, stdin=None, **kwargs):
        output_holder["cmd"] = cmd
        proc = MagicMock()
        proc.stdin = io.BytesIO()  # capture frames written
        proc.wait = MagicMock(return_value=0)
        proc.kill = MagicMock()
        return proc
    return factory


@pytest.mark.unit
def test_recorder_builds_expected_cmd(tmp_path) -> None:
    holder: dict = {}
    rec = FfmpegRecorder(video_size=(320, 240), framerate=10, crf=23, ffmpeg_binary="/usr/bin/ffmpeg")
    with patch("subprocess.Popen", side_effect=_mock_popen(holder)):
        rec.start_episode(tmp_path / "ep.mp4")
    cmd = holder["cmd"]
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert "-video_size" in cmd and "320x240" in cmd
    assert "-framerate" in cmd and "10" in cmd
    assert "-crf" in cmd and "23" in cmd
    assert cmd[-1] == str(tmp_path / "ep.mp4")
    assert "-pixel_format" in cmd and "rgb24" in cmd
    assert "libx264" in cmd


@pytest.mark.unit
def test_recorder_tee_receives_frames_from_both_channels(tmp_path) -> None:
    rec = FfmpegRecorder(video_size=(320, 240), ffmpeg_binary="/usr/bin/ffmpeg")
    with patch("subprocess.Popen", side_effect=_mock_popen({})):
        rec.start_episode(tmp_path / "ep.mp4")
        rec.start_subtask(tmp_path / "st.mp4")
    rec.stdin.write(b"frame_bytes")

    # After the write, both channels should have captured the payload via the
    # fake BytesIO stdins. We grab them from the tee's internal pipe list.
    captured = [p.getvalue() for p in rec.stdin._pipes]
    assert len(captured) == 2
    for c in captured:
        assert c == b"frame_bytes"


@pytest.mark.unit
def test_recorder_starting_subtask_closes_prior_one(tmp_path) -> None:
    rec = FfmpegRecorder(video_size=(320, 240), ffmpeg_binary="/usr/bin/ffmpeg")
    with patch("subprocess.Popen", side_effect=_mock_popen({})):
        rec.start_episode(tmp_path / "ep.mp4")
        rec.start_subtask(tmp_path / "st1.mp4")
        first = rec._subtask
        rec.start_subtask(tmp_path / "st2.mp4")

    # Old subtask handle replaced and its proc.wait called (closed).
    assert rec._subtask is not None
    assert rec._subtask.path.name == "st2.mp4"
    first.proc.wait.assert_called()


@pytest.mark.unit
def test_recorder_wait_closes_all(tmp_path) -> None:
    rec = FfmpegRecorder(video_size=(320, 240), ffmpeg_binary="/usr/bin/ffmpeg")
    with patch("subprocess.Popen", side_effect=_mock_popen({})):
        rec.start_episode(tmp_path / "ep.mp4")
        rec.start_subtask(tmp_path / "st.mp4")

    rec.wait(timeout=1.0)
    assert rec._episode is None
    assert rec._subtask is None


@pytest.mark.unit
def test_recorder_bool_is_always_true_for_base_task_compat() -> None:
    """Base_Task's `_del_eval_video_ffmpeg` checks `if self.eval_video_ffmpeg:`.
    A recorder that returns False there would skip cleanup, leaving subprocesses."""
    rec = FfmpegRecorder(video_size=(320, 240), ffmpeg_binary="/usr/bin/ffmpeg")
    assert bool(rec) is True
