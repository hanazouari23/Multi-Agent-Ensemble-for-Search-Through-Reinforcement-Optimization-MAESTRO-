"""
Train a Discrete CQL policy inside an experiment directory.

Expects an experiment manifest with ``artifacts.dataset`` pointing to the
HDF5 dataset produced by ``src/main.py``. Writes ``model.d3`` and updates
the manifest with ``artifacts.model``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import d3rlpy
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.preprocessing import StandardObservationScaler

src_path = Path(__file__).resolve().parent
root_path = src_path.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))

try:
    from .experiments import require_experiment_dir, log_artifact
except ImportError:
    from experiments import require_experiment_dir, log_artifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Discrete CQL policy from an experiment dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing experiment.json.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override the dataset path from the experiment manifest.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device passed to d3rlpy (e.g. cuda:0 or cpu:0).",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    experiment = require_experiment_dir(experiment_dir)

    logger.info("Experiment: %s (%s)", experiment.slug, experiment.purpose)

    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        dataset_filename = experiment.artifacts.get("dataset")
        if not dataset_filename:
            raise FileNotFoundError(
                "No dataset artifact recorded in the experiment manifest. "
                "Run src/main.py first or pass --dataset."
            )
        dataset_path = experiment_dir / dataset_filename

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    model_path = experiment_dir / "model.d3"
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from: %s", dataset_path)
    with dataset_path.open("rb") as f:
        dataset = ReplayBuffer.load(f, InfiniteBuffer())

    cql = DiscreteCQLConfig(
        batch_size=256,
        learning_rate=1e-4,
        gamma=0.99,
        target_update_interval=2_000,
        alpha=0.1,
        n_critics=2,
        observation_scaler=StandardObservationScaler(),
        encoder_factory=VectorEncoderFactory(
            hidden_units=[512, 512, 256],
            activation="relu",
        ),
    ).create(device=args.device)

    def save_epoch_checkpoint(algo, epoch: int, total_step: int) -> None:
        """Save a full model checkpoint after each epoch."""
        epoch_path = checkpoint_dir / f"model_epoch_{epoch}.d3"
        algo.save(str(epoch_path))
        logger.info("Epoch %02d (step %d) -> saved %s", epoch, total_step, epoch_path)

    history = cql.fit(
        dataset,
        n_steps=100_000,
        n_steps_per_epoch=10_000,
        experiment_name=f"discrete_cql_offline_{experiment.slug}",
        show_progress=True,
        save_interval=1,
        epoch_callback=save_epoch_checkpoint,
    )

    metrics_path = checkpoint_dir / "training_metrics.csv"
    if history:
        fieldnames = ["epoch"] + sorted(history[0][1].keys())
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for epoch, metrics in history:
                writer.writerow({"epoch": epoch, **metrics})
        logger.info("Saved per-epoch training metrics to: %s", metrics_path)
        logger.info("Metrics: %s", ", ".join(fieldnames[1:]))

    cql.save(str(model_path))
    log_artifact(experiment.slug, "model", "model.d3")

    logger.info("Saved trained policy to: %s", model_path)
    logger.info("Per-epoch checkpoints are in: %s", checkpoint_dir)


if __name__ == "__main__":
    main()
