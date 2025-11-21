from datetime import datetime
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import typer
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation

from pacman.agent import PacManAgent
from pacman.core import settings
from pacman.utils import HWNObservation, parse_duration, rmtree

# Register ale-py in Gymnasium API to use Ms. Pac-Man environment.
gym.register_envs(ale_py)

app = typer.Typer()

TRAIN_OUTPUT_PATH: Path = typer.Option(
    ".", "-o", "--output", help="Output path for training results (logs, model and data for plots)."
)
TRAINED_AGENT_PATH: Path = typer.Option(
    ..., "-p", "--model-path", help="Trained DQN agent (.h5 extension) file path."
)
PLOTS_DATA_PATH: Path = typer.Option(..., "--data-path", help="Training results data file.")
TRAINING_DATA_DURATION: str = typer.Option(
    "7d", "--duration", help="Duration for cleaning up old training results (eg. 1m, 1h30, etc.)."
)


@app.command()
def train(output: Path = TRAIN_OUTPUT_PATH) -> None:
    """Train the Ms. Pac-Man agent."""
    env = gym.make(
        "ALE/MsPacman-v5", frameskip=1, repeat_action_probability=settings.REPEAT_ACTION_PROBABILITY
    )
    env = AtariPreprocessing(env, frame_skip=settings.FRAME_SKIP)
    env = FrameStackObservation(env, settings.FRAME_STACK_SIZE)
    env = TransformObservation(env, lambda obs: obs.astype(np.float32) / 255.0, None)
    env = HWNObservation(env)
    agent = PacManAgent(env)
    agent.train(output=output)


@app.command()
def validate(path: Path = TRAINED_AGENT_PATH) -> None:
    """Run the trained agent in the Atari environment."""


@app.command()
def plot(path: Path = PLOTS_DATA_PATH) -> None:
    """Plot training results from the specified file."""


@app.command()
def cleanup(path: Path = TRAIN_OUTPUT_PATH, duration: str = TRAINING_DATA_DURATION) -> None:
    """Clean up training results in the specified output path older than given duration."""
    td = parse_duration(duration)
    now = datetime.now()
    removed = 0
    for d in [d for d in path.iterdir() if d.is_dir() and d.name.startswith("training")]:
        ts = datetime.strptime(d.name.replace("training-", ""), "%Y%m%d%H%M%S")
        if now - ts > td:
            rmtree(d)
            removed += 1
    if not removed:
        typer.secho("No old training directories found to remove.", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"Removed {removed} old training directories.", fg=typer.colors.GREEN)
