import json
from typing import Literal


class Settings:
    LEARNING_RATE: float = 1e-4
    DISCOUNT_FACTOR: float = 0.99
    # Soft update factor for target network. If TAU=1.0, it's equivalent to hard update.
    TAU: float = 1.0

    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.1
    EPSILON_DECAY_STEPS: int = 500_000
    EPSILON_DECAY_MODE: Literal["lin", "exp"] = "lin"

    BATCH_SIZE: int = 32
    MIN_REPLAY_MEMORY_SIZE: int = 50_000
    REPLAY_MEMORY_SIZE: int = 150_000

    TOTAL_EPISODES: int = 2000
    TRAIN_FREQ: int = 4
    NETWORK_SYNC_FREQ: int = 1000
    CHECKPOINT_FREQ: int = 100

    FRAME_SKIP: int = 4
    REPEAT_ACTION_PROBABILITY: float = 0
    FRAME_STACK_SIZE: int = 4
    NOOP_MAX: int = 30

    # Whether to normalize rewards (all rewards are divided by 10).
    # Also applies to death penalty.
    REWARD_NORMALIZATION: bool = True
    DEATH_PENALTY: float = -150.0

    def jsonify(self, indent: int = 2) -> str:
        return json.dumps({k: getattr(self, k) for k in dir(self) if k.isupper()}, indent=indent)


settings = Settings()
