from pathlib import Path
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

# ---- Offline training ----
cql.fit(
    dataset,
    n_steps=100_000,
    n_steps_per_epoch=10_000,
    experiment_name="discrete_cql_offline",
    show_progress=True,
)

# ---- Save model and configuration together ----
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
cql.save(str(MODEL_PATH))

print(f"Saved trained policy to: {MODEL_PATH}")