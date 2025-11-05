from typing import Literal


class Settings:
    LEARNING_RATE: float = 0.001
    DISCOUNT_FACTOR: float = 0.99
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_DECAY_STEPS: int = 200_000
    EPSILON_DECAY_MODE: Literal["lin", "exp"] = "lin"
    BATCH_SIZE: int = 64
    REPLAY_MEMORY_SIZE: int = 50_000
    TOTAL_EPISODES: int = 1000
    NETWORK_SYNC_FREQ: int = 1000
    FRAME_SKIP: int = 4
    FRAME_STACK_SIZE: int = 4


settings = Settings()
