# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 1psconstructor
from __future__ import annotations

import fcntl
import logging
import os
import struct
import time

import numpy as np

log = logging.getLogger(__name__)

# struct v4l2_format is 208 bytes on 64-bit; the union starts at offset 8.
VIDIOC_S_FMT = 0xC0D05605
V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_FIELD_NONE = 1
V4L2_COLORSPACE_SRGB = 8
V4L2_PIX_FMT_YUYV = int.from_bytes(b"YUYV", "little")

SCALE = 2
TARGET_P99 = 230.0
MAX_GAIN = 4.0
GAMMA = 0.7
REOPEN_INTERVAL_S = 5.0


class V4l2Sink:
    """Writes the IR picture into a v4l2loopback device as YUYV.

    The ET5 picture is 280x280 and dark: it is scaled up 2x and brightened so
    face recognition (Howdy) finds a face in it and does not drop it as too
    dark."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._fd: int | None = None
        self._size: tuple[int, int] | None = None
        self._failed_at: float | None = None

    def _open(self, width: int, height: int) -> bool:
        if (self._failed_at is not None
                and time.monotonic() - self._failed_at < REOPEN_INTERVAL_S):
            return False
        fd = None
        try:
            fd = os.open(self._path, os.O_WRONLY)
            fmt = bytearray(208)
            struct.pack_into("<I", fmt, 0, V4L2_BUF_TYPE_VIDEO_OUTPUT)
            struct.pack_into("<8I", fmt, 8, width, height, V4L2_PIX_FMT_YUYV,
                             V4L2_FIELD_NONE, width * 2, width * height * 2,
                             V4L2_COLORSPACE_SRGB, 0)
            fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)
        except OSError as exc:
            if self._failed_at is None:
                log.warning("IR mirror to %s unavailable: %s", self._path, exc)
            self._failed_at = time.monotonic()
            if fd is not None:
                os.close(fd)
            return False
        self._fd = fd
        self._size = (width, height)
        self._failed_at = None
        log.info("IR mirror writing %dx%d YUYV to %s", width, height, self._path)
        return True

    def write(self, picture: np.ndarray) -> None:
        y = picture.repeat(SCALE, axis=0).repeat(SCALE, axis=1)
        height, width = y.shape
        if self._fd is None or self._size != (width, height):
            self.close()
            if not self._open(width, height):
                return

        p99 = float(np.percentile(picture[::4, ::4], 99)) or 1.0
        gain = min(max(TARGET_P99 / p99, 1.0), MAX_GAIN)
        lut = (np.clip(np.arange(256) * gain, 0, 255) / 255.0) ** GAMMA * 255.0
        y = lut.astype(np.uint8)[y]

        yuyv = np.full((height, width * 2), 128, dtype=np.uint8)
        yuyv[:, 0::2] = y
        try:
            os.write(self._fd, yuyv.tobytes())
        except OSError as exc:
            log.warning("IR mirror write to %s failed: %s", self._path, exc)
            self.close()
            self._failed_at = time.monotonic()

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._fd = None
        self._size = None
