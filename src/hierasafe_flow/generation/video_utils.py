from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imageio.v3 as iio
import numpy as np

from hierasafe_flow.utils.io import ensure_dir


def save_video_frames(frames: Sequence, path: str | Path, fps: int = 16) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    arrays = [_frame_to_array(frame) for frame in frames]
    iio.imwrite(target, arrays, fps=fps)


def _frame_to_array(frame) -> np.ndarray:
    if hasattr(frame, "convert"):
        return np.asarray(frame.convert("RGB"))
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return array

