import logging
from datetime import datetime
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np
import tensorflow as tf

from pacman.core import (
    EpsilonGreedy,
    Experience,
    PacManDQN,
    ReplayMemory,
    TrainingStats,
    add_file_handler,
    settings,
)

logger = logging.getLogger(__name__)


class PacManAgent:
    def __init__(self, env: gym.Env) -> None:
        self._env = env
        self._epsilon = EpsilonGreedy(
            settings.EPSILON_START,
            settings.EPSILON_END,
            settings.EPSILON_DECAY_STEPS,
            settings.EPSILON_DECAY_MODE,
        )
        self._memory = ReplayMemory(settings.REPLAY_MEMORY_SIZE)

        self._online_network = PacManDQN(
            env.action_space.n,  # noqa: F821
            env.observation_space.shape,
        )
        self._target_network = PacManDQN(
            env.action_space.n,  # noqa: F821
            env.observation_space.shape,
        )
        self._online_network.compile(
            loss="mse", optimizer=tf.keras.optimizers.Adam(settings.LEARNING_RATE)
        )

        dummy = tf.random.normal((1,) + self._env.observation_space.shape)
        self._online_network.predict(dummy, verbose=0)
        self._target_network.predict(dummy, verbose=0)
        self._target_network.set_weights(self._online_network.get_weights())

    def load(self, model: Path) -> Self:
        """Load a trained agent from a file."""
        if not model.name.lower().endswith(".weights.h5"):
            raise ValueError("Model file with weights must end with '.weights.h5'")
        self._online_network.load_weights(model)

    def _warmup_replay_memory(self) -> None:
        state = self._env.reset()[0]
        while len(self._memory) < settings.MIN_REPLAY_MEMORY_SIZE:
            action = self._env.action_space.sample()
            next_state, reward, done, _, _ = self._env.step(action)
            experience = Experience(state, action, float(reward), next_state, done)
            self._memory.push(experience)
            state = next_state if not done else self._env.reset()[0]

    def _soft_update_target_network(self) -> None:
        """Soft update target network weights from the online network."""
        online_weights = self._online_network.get_weights()
        target_weights = self._target_network.get_weights()
        updated_weights = [
            settings.TAU * ow + (1 - settings.TAU) * tw
            for ow, tw in zip(online_weights, target_weights, strict=True)
        ]
        self._target_network.set_weights(updated_weights)

    def _replay_experience(self) -> float:
        states, actions, rewards, next_states, dones = self._memory.sample(settings.BATCH_SIZE)
        current_qs = self._online_network.predict(states, verbose=0)
        target_qs = self._target_network.predict(next_states, verbose=0)
        for idx in range(settings.BATCH_SIZE):
            action, reward, done = actions[idx], rewards[idx], dones[idx]
            current_qs[idx][action] = (
                reward if done else reward + settings.DISCOUNT_FACTOR * np.max(target_qs[idx])
            )
        history = self._online_network.fit(
            states, current_qs, settings.BATCH_SIZE, epochs=1, verbose=0
        )
        return history.history["loss"][0]

    def act(self, state: np.ndarray, step: int) -> int:
        """Select an action for the given state using epsilon-greedy policy."""
        if np.random.rand() < self._epsilon.value(step):
            return self._env.action_space.sample()
        qs = self._online_network.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(qs[0])

    def _summarize_episode(self, stats: TrainingStats, rewards: list[float], output: Path) -> None:
        with (output / "training-stats.jsonl").open("a") as f:
            f.write(f"{stats.jsonify()}\n")
            f.flush()

        if stats.episode % settings.CHECKPOINT_FREQ == 0:
            logger.info("Saving model checkpoint at episode %d", stats.episode)
            directory = output / "checkpoints"
            if not directory.exists():
                directory.mkdir()
            self._online_network.save_weights(directory / f"agent-ep{stats.episode}.weights.h5")

        if stats.episode % 10 == 0 or stats.episode == 1:
            logger.info(
                "Ep %d | R: %6.1f | Avg R: %6.1f | e: %.3f | Step: %d",
                stats.episode,
                rewards[-1],
                np.mean(rewards[-10:]),
                self._epsilon.value(stats.step),
                stats.step,
            )

    def _train_over_episodes(self, output: Path) -> None:
        step = 1
        rewards = []
        for episode in range(1, settings.TOTAL_EPISODES + 1):
            state = self._env.reset()[0]
            done = False
            total = 0.0
            loss = 0.0
            while not done:
                action = self.act(state, step=step)
                next_state, reward, done, _, _ = self._env.step(action)
                reward = float(reward)
                total += reward
                self._memory.push(Experience(state, action, reward, next_state, done))

                if (
                    len(self._memory) >= settings.MIN_REPLAY_MEMORY_SIZE
                    and step % settings.TRAIN_FREQ == 0
                ):
                    loss = self._replay_experience()

                if step % settings.NETWORK_SYNC_FREQ == 0:
                    self._soft_update_target_network()
                step += 1

            rewards.append(total)
            stats = TrainingStats(
                episode=episode,
                step=step,
                epsilon=self._epsilon.value(step),
                reward=total,
                loss=loss,
            )
            self._summarize_episode(stats, rewards, output)

    def train(self, output: Path) -> None:
        """Train the agent and save results to the output path."""
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        output = output / f"training-{ts}"
        logger.info("Creating output directory for training results: %s", output)
        output.mkdir()
        add_file_handler(output / "training.log", logger=logger)

        logger.info("Saving training settings to: %s", output / "settings.json")
        with open(output / "settings.json", "w") as f:
            f.write(settings.jsonify())

        start = datetime.now()
        logger.info("Training started, total episodes: %d", settings.TOTAL_EPISODES)
        logger.info("Warming up replay memory (min size: %d)", settings.MIN_REPLAY_MEMORY_SIZE)
        self._warmup_replay_memory()

        logger.info("Starting main training loop with %d experiences", len(self._memory))
        self._train_over_episodes(output)

        logger.info(
            "Training completed, saving final model to: %s", output / "agent-final.weights.h5"
        )
        self._online_network.save_weights(output / "agent-final.weights.h5")
        logger.info("Total training time: %s", datetime.now() - start)

    def validate(self, episodes: int = 1) -> None:
        """Run the trained agent in the Atari environment."""
        logger.info("Starting validation for %d episodes", episodes)
        for _ in range(episodes):
            state = self._env.reset()[0]
            done = False
            total = 0
            while not done:
                action = np.argmax(
                    self._online_network.predict(np.expand_dims(state, axis=0), verbose=0)[0]
                )
                state, reward, done, _, _ = self._env.step(action)
                total += reward
            logger.info("Validation episode completed with total reward: %d", total)
