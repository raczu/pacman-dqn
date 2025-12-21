from typing import override

import gymnasium as gym

from pacman.agents.base import Agent


class RandomAgent(Agent):
    AGENT_TYPE: str = "random"

    @override
    def act(self, state: gym.Env) -> int:
        """Always select a random action."""
        return self._env.action_space.sample()
