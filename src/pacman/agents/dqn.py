from typing import override

import numpy as np

from pacman.agents.base import TrainableAgent
from pacman.core import settings


class DQNAgent(TrainableAgent):
    @override
    def _compute_targets(
        self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray
    ) -> np.ndarray:
        """Compute target Q-values using the DQN update rule."""
        next_qs_target = self._target_network.predict(next_states, verbose=0)
        max_next_qs = np.max(next_qs_target, axis=1)
        targets = rewards + settings.DISCOUNT_FACTOR * max_next_qs * (1 - dones.astype(np.float32))
        return targets


class DDQNAgent(TrainableAgent):
    @override
    def _compute_targets(
        self, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray
    ) -> np.ndarray:
        """Compute target Q-values using the DDQN update rule."""
        next_qs_target = self._target_network.predict(next_states, verbose=0)
        next_qs_online = self._online_network.predict(next_states, verbose=0)
        best_actions = np.argmax(next_qs_online, axis=1)
        target_qs = next_qs_target[np.arange(len(next_qs_target)), best_actions]
        targets = rewards + settings.DISCOUNT_FACTOR * target_qs * (1 - dones.astype(np.float32))
        return targets
