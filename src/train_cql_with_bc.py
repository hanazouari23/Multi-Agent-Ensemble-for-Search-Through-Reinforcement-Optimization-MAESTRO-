"""Train DiscreteCQL for query routing, optionally warm-started from BC weights.

The dataset is a tree-branch MDP dataset. If a BC checkpoint is provided,
the Q-network encoder is initialized from the BC policy weights so CQL
starts from a reasonable policy instead of random weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from d3rlpy.models.encoders import VectorEncoderFactory

try:
    from .train_bc import BCPolicy
except ImportError:
    from train_bc import BCPolicy


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_H5 = (
    PROJECT_ROOT / "outputs" / "mdp_dataset_tree_branches_5000_terminal_reward.h5"
)
DEFAULT_BC_POLICY = PROJECT_ROOT / "outputs" / "bc_checkpoints" / "bc_policy.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cql_checkpoints_bc_warmstart"

STATE_DIM = 399
N_ACTIONS = 4
HIDDEN_UNITS = [512, 512, 256]


def build_cql(
    learning_rate: float = 1e-4,
    batch_size: int = 256,
    alpha: float = 1.0,
    gamma: float = 0.99,
    n_critics: int = 1,
    target_update_interval: int = 8000,
) -> DiscreteCQL:
    """Create a DiscreteCQL algorithm with the same encoder as BC."""
    config = DiscreteCQLConfig(
        learning_rate=learning_rate,
        encoder_factory=VectorEncoderFactory(
            hidden_units=HIDDEN_UNITS, activation="relu"
        ),
        batch_size=batch_size,
        gamma=gamma,
        alpha=alpha,
        n_critics=n_critics,
        target_update_interval=target_update_interval,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return DiscreteCQL(config=config, device=device, enable_ddp=False)


def load_bc_state_dict(policy_path: Path) -> dict[str, torch.Tensor]:
    """Load the BC model and return its state dict."""
    model = BCPolicy(STATE_DIM, N_ACTIONS, HIDDEN_UNITS)
    model.load_state_dict(
        torch.load(policy_path, map_location="cpu", weights_only=True)
    )
    return model.state_dict()


def map_bc_to_cql_state(bc_state: dict[str, torch.Tensor], n_critics: int) -> dict[str, torch.Tensor]:
    """Map BC MLP keys to d3rlpy DiscreteCQL Q-function keys.

    BC: net.0, net.2, net.4 are encoder layers; net.6 is the output fc.
    CQL q_function: {critic_idx}._encoder._layers.{idx} and {critic_idx}._fc.
    """
    mapping = {
        "net.0.weight": "_encoder._layers.0.weight",
        "net.0.bias": "_encoder._layers.0.bias",
        "net.2.weight": "_encoder._layers.2.weight",
        "net.2.bias": "_encoder._layers.2.bias",
        "net.4.weight": "_encoder._layers.4.weight",
        "net.4.bias": "_encoder._layers.4.bias",
        "net.6.weight": "_fc.weight",
        "net.6.bias": "_fc.bias",
    }

    cql_state: dict[str, torch.Tensor] = {}
    for critic_idx in range(n_critics):
        for bc_key, cql_key in mapping.items():
            new_key = f"{critic_idx}.{cql_key}"
            cql_state[new_key] = bc_state[bc_key]

    return cql_state


def initialize_cql_from_bc(cql: DiscreteCQL, bc_policy_path: Path) -> None:
    """Copy BC encoder weights into CQL Q-network(s) and target Q-network(s)."""
    bc_state = load_bc_state_dict(bc_policy_path)

    # Determine number of critics from the q_function state dict.
    q_func_state = cql.impl.q_function.state_dict()
    n_critics = len({k.split(".")[0] for k in q_func_state.keys()})

    cql_state = map_bc_to_cql_state(bc_state, n_critics)

    # Load into online Q-network.
    cql.impl.q_function.load_state_dict(cql_state, strict=False)

    # Sync target Q-network.
    if hasattr(cql.impl, "update_target"):
        cql.impl.update_target()

    print(f"Initialized CQL Q-network(s) from BC checkpoint: {bc_policy_path}")


def save_checkpoint(cql: DiscreteCQL, output_dir: Path, epoch: int) -> Path:
    """Save a full d3rlpy algorithm checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"discrete_cql_policy_epoch_{epoch}.d3"
    cql.save(str(checkpoint_path))
    return checkpoint_path


def train(
    dataset_h5: Path,
    output_dir: Path,
    bc_policy_path: Path | None = None,
    n_epochs: int = 10,
    steps_per_epoch: int = 1000,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    alpha: float = 1.0,
    gamma: float = 0.99,
    n_critics: int = 1,
    save_interval: int = 1,
) -> None:
    """Train DiscreteCQL with optional BC warm-start."""
    print(f"Loading dataset: {dataset_h5}")
    with dataset_h5.open("rb") as f:
        replay_buffer = ReplayBuffer.load(f, InfiniteBuffer())

    # d3rlpy 2.x fit API expects the ReplayBuffer as the dataset.
    dataset = replay_buffer
    print(f"Episodes: {len(dataset.episodes):,}")

    print("Building CQL")
    cql = build_cql(
        learning_rate=learning_rate,
        batch_size=batch_size,
        alpha=alpha,
        gamma=gamma,
        n_critics=n_critics,
    )
    cql.build_with_dataset(dataset)

    if bc_policy_path is not None and bc_policy_path.is_file():
        initialize_cql_from_bc(cql, bc_policy_path)
    elif bc_policy_path is not None:
        print(f"BC checkpoint not found, training from scratch: {bc_policy_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata.
    meta = {
        "dataset_h5": str(dataset_h5),
        "bc_policy_path": str(bc_policy_path) if bc_policy_path else None,
        "n_epochs": n_epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "alpha": alpha,
        "gamma": gamma,
        "n_critics": n_critics,
        "state_dim": STATE_DIM,
        "n_actions": N_ACTIONS,
        "hidden_units": HIDDEN_UNITS,
    }
    with (output_dir / "training_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    total_steps = n_epochs * steps_per_epoch
    checkpoint_paths: list[Path] = []

    # d3rlpy 2.x fit returns a generator of epochs. We iterate and save checkpoints.
    for epoch, metrics in enumerate(
        cql.fit(
            dataset,
            n_steps=total_steps,
            n_steps_per_epoch=steps_per_epoch,
            experiment_name=None,
            with_timestamp=False,
        ),
        start=1,
    ):
        print(f"Epoch {epoch:02d}: {metrics}")
        if epoch % save_interval == 0 or epoch == n_epochs:
            ckpt = save_checkpoint(cql, output_dir, epoch)
            checkpoint_paths.append(ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    print(f"\nTraining complete. Saved {len(checkpoint_paths)} checkpoints to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DiscreteCQL with optional BC warm-start")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bc-policy", type=Path, default=DEFAULT_BC_POLICY)
    parser.add_argument("--no-bc-warmstart", action="store_true", help="Train CQL from scratch")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-critics", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    args = parser.parse_args()

    train(
        dataset_h5=args.dataset,
        output_dir=args.output_dir,
        bc_policy_path=None if args.no_bc_warmstart else args.bc_policy,
        n_epochs=args.n_epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        gamma=args.gamma,
        n_critics=args.n_critics,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
