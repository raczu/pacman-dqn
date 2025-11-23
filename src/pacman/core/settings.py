import json
from typing import Literal


class Settings:
    LEARNING_RATE: float = 1e-3
    DISCOUNT_FACTOR: float = 0.99
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.01
    EPSILON_DECAY_STEPS: int = 200_000
    EPSILON_DECAY_MODE: Literal["lin", "exp"] = "exp"
    BATCH_SIZE: int = 64
    MIN_REPLAY_MEMORY_SIZE: int = 1000
    REPLAY_MEMORY_SIZE: int = 80_000
    TOTAL_EPISODES: int = 1000
    TRAIN_FREQ: int = 5
    NETWORK_SYNC_FREQ: int = 50
    CHECKPOINT_FREQ: int = 100
    TAU: float = 1.0
    FRAME_SKIP: int = 4
    REPEAT_ACTION_PROBABILITY: float = 0
    FRAME_STACK_SIZE: int = 4

    def jsonify(self, indent: int = 2) -> str:
        return json.dumps({k: getattr(self, k) for k in dir(self) if k.isupper()}, indent=indent)


settings = Settings()
