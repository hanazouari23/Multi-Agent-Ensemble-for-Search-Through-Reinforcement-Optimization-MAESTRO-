from pathlib import Path
import csv

import d3rlpy
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.preprocessing import StandardObservationScaler
# ---- Paths ----
# DATASET_EXPERT_PATH = Path("../outputs/mdp_dataset_trajectories_expert_5k.h5")
# DATASET_RANDOM_PATH = Path("../outputs/mdp_dataset_trajectories_random_5k.h5")
# DATASET_PATH = Path("../outputs/mdp_dataset_trajectories_two_policies_5k.h5")
# MODEL_PATH = Path("../outputs/discrete_cql_policy.d3")

# src/train_cql.py -> project root is one directory above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "mdp_dataset_tree_branches_5000.h5"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "discrete_cql_policy_after_tree.d3"
)

# Directory where per-epoch checkpoints will be saved.
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "cql_checkpoints_after_tree"

print(f"Loading dataset from: {DATASET_PATH}")

if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Dataset file does not exist: {DATASET_PATH}")
# ---- Load d3rlpy HDF5 dataset ----
# In d3rlpy 2.x, HDF5 data is restored as a ReplayBuffer.
with DATASET_PATH.open("rb") as f:
    dataset = ReplayBuffer.load(f, InfiniteBuffer())

# Optional sanity checks: your observation vector should have 399 features.
# print("Observation shape:", dataset.transition_picker.observation_signature.shape)
# print("Action size:", dataset.dataset_info.action_size)
# print("Action space:", dataset.dataset_info.action_space)

# ---- Create discrete CQL ----
# Use device="cuda:0" if CUDA/PyTorch GPU support is available.
# cql = DiscreteCQLConfig(
#     batch_size=256,
#     learning_rate=6.25e-5,
#     target_update_interval=8_000,
# ).create(device="cpu:0")


cql = DiscreteCQLConfig(
    batch_size=256,
    learning_rate=1e-4,
    gamma=0.99,
    target_update_interval=2_000,
    alpha=0.1,  # tune; start less conservative than the usual default
    n_critics=2,
    observation_scaler=StandardObservationScaler(),
    encoder_factory=VectorEncoderFactory(
        hidden_units=[512, 512, 256],
        activation="relu",
    ),
).create(device="cuda:0")

# ---- Per-epoch checkpoint callback ----
# d3rlpy already saves internal parameters with save_interval=1, but this
# callback writes an explicit, full-model checkpoint at the end of every
# epoch so you can pick the best early-stopping checkpoint later.
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def save_epoch_checkpoint(algo, epoch: int, total_step: int) -> None:
    """Save a full model checkpoint after each epoch."""
    checkpoint_path = CHECKPOINT_DIR / f"discrete_cql_policy_after_tree_epoch_{epoch}.d3"
    algo.save(str(checkpoint_path))
    print(f"Epoch {epoch:02d} (step {total_step:,}) -> saved {checkpoint_path}")


# ---- Offline training ----
# fit returns a list of (epoch, metrics_dict) tuples, which we capture below.
history = cql.fit(
    dataset,
    n_steps=100_000,
    n_steps_per_epoch=10_000,
    experiment_name="discrete_cql_offline",
    show_progress=True,
    save_interval=1,
    epoch_callback=save_epoch_checkpoint,
)

# ---- Save training metrics per epoch ----
# This makes it easy to correlate training loss curves with test-set
# performance when picking the best epoch checkpoint.
METRICS_PATH = CHECKPOINT_DIR / "training_metrics.csv"
if history:
    fieldnames = ["epoch"] + sorted(history[0][1].keys())
    with METRICS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, metrics in history:
            writer.writerow({"epoch": epoch, **metrics})
    print(f"Saved per-epoch training metrics to: {METRICS_PATH}")
    print("Metrics:", ", ".join(fieldnames[1:]))

# ---- Save final model and configuration together ----
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
cql.save(str(MODEL_PATH))

print(f"Saved trained policy to: {MODEL_PATH}")
print(f"Per-epoch checkpoints are in: {CHECKPOINT_DIR}")
