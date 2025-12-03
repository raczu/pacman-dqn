import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import ale_py
import gymnasium as gym
import typer
from gymnasium.wrappers import RecordVideo

from pacman.agents import AgentFactory
from pacman.core import TrainingStatsHistory, TrainingStepStats, configure_logger
from pacman.utils import (
    make_env,
    parse_duration,
    rmtree,
    save_epsilon_decay_plot,
    save_learning_curve_plot,
    save_loss_curve_plot,
)

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
TRAIN_AGENT_TYPE: Literal["dqn", "ddqn"] = typer.Option(
    "dqn", "--agent", help="Type of agent to train."
)
TRAINED_AGENT_PATH: Path | None = typer.Option(
    None,
    "--model-path",
    help="Trained DQN agent (.h5 extension) file path. Not required for random agent.",
)
VALIDATION_VIDEO_PATH: Path = typer.Option(
    ".", "--video-path", help="Output directory for validation videos."
)
VALIDATION_VIDEO_EPISODES: int = typer.Option(
    1, "--episodes", help="Number of episodes to record during validation."
)
VALIDATION_AGENT_TYPE: Literal["dqn", "ddqn", "random"] = typer.Option(
    "dqn", "--agent", help="Type of agent to validate."
)
PLOTS_DATA_PATHS: list[Path] = typer.Option(
    ..., "--data-path", help="Training results data file(s) (.jsonl extension)."
)
PLOT_AVERAGE_WINDOW: int = typer.Option(
    100, "--avg-window", help="Moving average window size for learning curve."
)
PLOTS_OUTPUT_PATH: Path = typer.Option(".", "-o", "--output", help="Output directory for plots.")
TRAINING_DATA_DURATION: str = typer.Option(
    "7d", "--duration", help="Duration for cleaning up old training results (eg. 1m, 1h30, etc.)."
)


@app.command()
def train(
    output: Path = TRAIN_OUTPUT_PATH, agent_type: Literal["dqn", "ddqn"] = TRAIN_AGENT_TYPE
) -> None:
    """Train the Ms. Pac-Man agent."""
    env = make_env()
    agent = AgentFactory.create(agent_type, env)
    agent.train(output=output)
    env.close()


@app.command()
def validate(
    path: Path | None = TRAINED_AGENT_PATH,
    agent_type: Literal["dqn", "ddqn", "random"] = VALIDATION_AGENT_TYPE,
    output: Path = VALIDATION_VIDEO_PATH,
    episodes: int = VALIDATION_VIDEO_EPISODES,
) -> None:
    """Run the agent in the Atari environment."""
    if agent_type != "random" and path is None:
        typer.secho(
            "A trained agent model path must be provided for DQN or DDQN agents.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if episodes > 1:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        output = output / f"validation-{ts}"
        typer.secho(
            "Creating output directory for validation videos: {output} as there are multiple episodes to record",
            fg=typer.colors.BLUE,
        )
        output.mkdir()

    env = make_env()
    env = RecordVideo(
        env, video_folder=str(output), name_prefix="pacman-agent", episode_trigger=lambda x: True
    )
    agent = AgentFactory.create(agent_type, env)
    if path is not None:
        agent.load(path)
    agent.validate(episodes=episodes)
    env.close()
    typer.secho(f"Validation video(s) saved to: {output}", fg=typer.colors.GREEN)


@app.command()
def plot(
    paths: list[Path] = PLOTS_DATA_PATHS,
    output: Path = PLOTS_OUTPUT_PATH,
    window: int = PLOT_AVERAGE_WINDOW,
) -> None:
    """Plot training results from the specified file(s)."""
    for path in paths:
        if not path.exists():
            typer.secho(f"Data file not found: {path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    output = output / f"plots-{ts}"
    typer.secho(f"Creating output directory for plots: {output}", fg=typer.colors.BLUE)
    output.mkdir()

    histories = []
    for path in paths:
        with path.open() as f:
            history = TrainingStatsHistory(
                stats=[TrainingStepStats(**json.loads(line)) for line in f]
            )
        if len(history.episodes) < window:
            typer.secho(
                f"Moving average window is larger than number of episodes in file: {path}",
                fg=typer.colors.RED,
            )
            typer.Exit(code=1)
        histories.append(history)

    save_learning_curve_plot(histories, window, output)
    save_epsilon_decay_plot(histories, output)
    save_loss_curve_plot(histories, window, output)


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
