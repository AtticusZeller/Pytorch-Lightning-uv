from pathlib import Path
from typing import Annotated

import typer

# Common options
ConfigPath = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
]

# Action options
EDAFlag = Annotated[bool, typer.Option("--eda", help="Explore dataset", is_flag=True)]

TrainFlag = Annotated[
    bool, typer.Option("--train", help="Train the model", is_flag=True)
]


# Sweep options
SweepFlag = Annotated[
    bool, typer.Option("--sweep", help="Run hyperparameter sweep", is_flag=True)
]

SweepConfigPath = Annotated[
    Path | None,
    typer.Option(
        "--sweep-config",
        help="Path to sweep configuration file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
]

SweepCount = Annotated[
    int, typer.Option("--sweep-count", help="Number of sweep runs", min=1)
]


# Evaluation options
EvalFlag = Annotated[
    str | None, typer.Option("--eval", help="WandB run ID for evaluation")
]
