import re
from datetime import timedelta
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    TransformObservation,
)

from pacman.core import settings

_DURATION_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>[smhdSMHD]s?)")
_UNIT_TO_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}


class HWCObservation(gym.ObservationWrapper):
    """Gym observation wrapper to convert observations to HWC format."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_shape = self.observation_space.shape
        if len(obs_shape) != 3:
            raise ValueError("Observation space must be 3-dimensional")
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[1], obs_shape[2], obs_shape[0]),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation: gym.core.ObsType) -> gym.core.ObsType:
        return observation.transpose(1, 2, 0)


def parse_duration(duration: str) -> timedelta:
    """Parse a duration string into a timedelta object.

    The duration string should be in the format like '1m30s', '1h', '2d', or '45s',
    where 'd' stands for days, 'h' for hours, 'm' for minutes, and 's' for seconds.
    """
    duration = duration.strip()
    if not duration:
        raise ValueError("Duration string is empty")
    if any(ch.isspace() for ch in duration):
        raise ValueError("Duration string must not contain whitespace")

    matches = list(_DURATION_RE.finditer(duration))
    if not matches:
        raise ValueError(f"Could not parse duration: '{duration}'")

    concatenated = "".join(m.group(0) for m in matches)
    if concatenated != duration:
        raise ValueError(f"Could duration format: '{duration}'")

    seconds = 0
    for m in matches:
        val = float(m.group("value"))
        unit = m.group("unit").lower()
        if unit not in _UNIT_TO_SECONDS:
            raise ValueError(f"Unknown time unit: '{unit}'")
        seconds += val * _UNIT_TO_SECONDS[unit]

    if seconds == 0:
        raise ValueError("Duration must be greater than zero")
    return timedelta(seconds=seconds)


def rmtree(directory: Path) -> None:
    """Recursively delete a directory and all its contents."""
    for root, dirs, files in directory.walk(top_down=False):
        for name in files:
            (root / name).unlink()
        for name in dirs:
            (root / name).rmdir()
    directory.rmdir()


def make_env() -> gym.Env:
    """Create and return a Gym environment for Ms. Pac-Man."""
    env = gym.make(
        "ALE/MsPacman-v5",
        frameskip=1,
        repeat_action_probability=settings.REPEAT_ACTION_PROBABILITY,
        render_mode="rgb_array",
    )
    env = AtariPreprocessing(env, frame_skip=settings.FRAME_SKIP)
    env = FrameStackObservation(env, settings.FRAME_STACK_SIZE)
    env = TransformObservation(env, lambda obs: obs.astype(np.float32) / 255.0, None)
    env = HWCObservation(env)
    return env
