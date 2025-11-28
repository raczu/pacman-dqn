import gymnasium as gym

from pacman.agents.base import Agent, TrainableAgent
from pacman.agents.dqn import DDQNAgent, DQNAgent
from pacman.agents.random import RandomAgent


class AgentFactory:
    _REGISTRY: dict[str, type[Agent | TrainableAgent]] = {
        "random": RandomAgent,
        "dqn": DQNAgent,
        "ddqn": DDQNAgent,
    }

    @classmethod
    def create(cls, agent: str, env: gym.Env) -> Agent | TrainableAgent:
        normalized = agent.lower()
        if normalized not in cls._REGISTRY:
            raise ValueError(f"Unknown agent type: {normalized}")
        return cls._REGISTRY[normalized](env)
