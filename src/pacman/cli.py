from datetime import datetime
from pathlib import Path

import ale_py
import gymnasium as gym
import typer
from gymnasium.wrappers import RecordVideo

from pacman.agent import PacManAgent
from pacman.core import configure_logger
from pacman.utils import make_env, parse_duration, rmtree

# Register ale-py in Gymnasium API to use Ms. Pac-Man environment.
gym.register_envs(ale_py)

configure_logger()

app = typer.Typer()

TRAIN_OUTPUT_PATH: Path = typer.Option(
    ".",
    "-o",
    "--output",
    help="Output directory for training results (logs, model and data for plots).",
)
TRAINED_AGENT_PATH: Path = typer.Option(
    ..., "--model-path", help="Trained DQN agent (.h5 extension) file path."
)
TRAINED_AGENT_VIDEO_PATH: Path = typer.Option(
    ".", "--video-path", help="Output directory for validation videos."
)
TRAINED_AGENT_VIDEO_EPISODES: int = typer.Option(
    1, "--episodes", help="Number of episodes to record during validation."
)
PLOTS_DATA_PATH: Path = typer.Option(..., "--data-path", help="Training results data file.")
TRAINING_DATA_DURATION: str = typer.Option(
    "7d", "--duration", help="Duration for cleaning up old training results (eg. 1m, 1h30, etc.)."
)


@app.command()
def train(output: Path = TRAIN_OUTPUT_PATH) -> None:
    """Train the Ms. Pac-Man agent."""
    env = make_env()
    agent = PacManAgent(env)
    agent.train(output=output)
    env.close()


@app.command()
def validate(
    path: Path = TRAINED_AGENT_PATH,
    output: Path = TRAINED_AGENT_VIDEO_PATH,
    episodes: int = TRAINED_AGENT_VIDEO_EPISODES,
) -> None:
    """Run the trained agent in the Atari environment."""
    env = make_env()
    env = RecordVideo(
        env, video_folder=str(output), name_prefix="pacman-agent", episode_trigger=lambda x: True
    )
    agent = PacManAgent(env)
    agent.load(path)
    agent.validate(episodes=episodes)
    env.close()
    typer.secho(f"Validation video saved to: {output}", fg=typer.colors.GREEN)


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
