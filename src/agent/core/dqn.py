import random
from collections import deque
from typing import Literal, NamedTuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


class Experience(NamedTuple):
    """
    Single experience for DQN replay memory.

    Attributes:
        state (np.ndarray): The state observed.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (np.ndarray): The next state observed after taking the action.
        terminated (bool): Whether the episode terminated after this experience.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool


class ReplayMemory:
    def __init__(self, maxlen: int) -> None:
        self.memory = deque(maxlen=maxlen)

    def push(self, experience: Experience) -> None:
        self.memory.append(experience)

    def sample(self, size: int) -> list[Experience]:
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, size)

    def __len__(self) -> int:
        return len(self.memory)


class PacManDQN(Model):
    def __init__(self, actions_num: int) -> None:
        super().__init__()
        self.conv1 = layers.Conv2D(32, (8, 8), strides=4, activation="relu")
        self.conv2 = layers.Conv2D(64, (4, 4), strides=2, activation="relu")
        self.conv3 = layers.Conv2D(64, (3, 3), strides=1, activation="relu")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(512, activation="relu")
        self.actions = layers.Dense(actions_num, activation="linear")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.actions(x)


class EpsilonGreedy:
    def __init__(
        self, start: float, end: float, steps: int, mode: Literal["lin", "exp"] = "lin"
    ) -> None:
        """
        Epsilon greedy strategy for action selection.

        Args:
            start: Initial epsilon value.
            end: Final epsilon value.
            steps: Number of steps to decay epsilon.
            mode: Decay mode, either linear ("lin") or exponential ("exp").
        """
        if mode not in ("lin", "exp"):
            raise ValueError("mode must be 'lin' or 'exp'")
        self.start = start
        self.end = end
        self.steps = steps
        self.mode = mode

    def value(self, step: int) -> float:
        """
        Calculate epsilon value based on the current step.

        Args:
            step: Current training step.
        """
        if step >= self.steps:
            return self.end
        if self.mode == "lin":
            frac = step / self.steps
            eps = self.start - frac * (self.start - self.end)
        else:
            rate = np.log(self.end / self.start) / self.steps
            eps = self.start * np.exp(rate * step)
        return eps
