from pathlib import Path

import typer

app = typer.Typer()

OUTPUT_PATH: Path = typer.Option(
    ".", "-o", "--output", help="Output path for training results (logs, model and plots)."
)
TRAINED_AGENT_PATH: Path = typer.Option(
    ..., "-p", "--model-path", help="Trained DQN Agent (.h5 extension) file path."
)
PLOTS_DATA_PATH: Path = typer.Option(..., "--data-path", help="Training results data file.")


@app.command()
def train(output: Path = OUTPUT_PATH) -> None:
    """Train the Ms. Pac-Man agent."""


@app.command()
def run(path: Path = TRAINED_AGENT_PATH) -> None:
    """Run the trained agent in the Atari environment."""


@app.command()
def plot(path: Path = PLOTS_DATA_PATH) -> None:
    """Plot training results from the specified file."""
