"""
Compare a saved RL policy to random actions on the same env.
Edit RL_ZIP_PATH and ALGORITHM at the top, then run (e.g. click Run). Paths are relative to this file's directory.
"""
import os
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import pandas as pd
import torch

from reaction_optimizer import (
    DEFAULT_DATA_FILE,
    DielsAlderOptEnv,
    ConditionAwareTransformer,
    DielsAlderTransformer,
    build_graph_from_row,
    load_stats,
    _resolve_run_paths,
    _match_node_attr_dim,
    _match_edge_attr_dim,
    _infer_edge_dim_from_state,
    _infer_node_dim_from_state,
    RL_ALGORITHMS,
)

# --- Config: set these then run ---
# Path relative to this file's directory, or absolute path. Works when you click Run.
RL_ZIP_PATH = "trainedmodels/default_td3_row0.zip"
ALGORITHM = "td3"

# Defaults (same as training script)
DATA_ROOT = _script_dir
MODEL_PATH = None
STATS_PATH = None
DATA_FILE = DEFAULT_DATA_FILE
ROW_IDX = 0
N_RANDOM = 30


def main():
    data_root = DATA_ROOT
    model_path, stats_path = _resolve_run_paths(data_root, MODEL_PATH, STATS_PATH)
    stats = load_stats(data_root, stats_path)
    ts_mean = float(stats["ts_mean"])
    ts_std = float(stats["ts_std"])
    feature_stats = stats["feature_stats"]

    data_path = data_root / DATA_FILE
    data_df = pd.read_csv(data_path)
    if ROW_IDX < 0 or ROW_IDX >= len(data_df):
        raise IndexError("row-idx is out of range.")
    row = data_df.iloc[ROW_IDX]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = None
    expected_edge_dim = None
    expected_node_dim = None
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        expected_edge_dim = _infer_edge_dim_from_state(state_dict)
        expected_node_dim = _infer_node_dim_from_state(state_dict)

    graph_data = build_graph_from_row(row, feature_stats, expected_node_dim, expected_edge_dim)
    graph_data = graph_data.to(device)
    edge_dim = graph_data.edge_attr.size(1)
    node_dim = graph_data.x.size(1)
    graph_data = _match_node_attr_dim(graph_data, node_dim)
    graph_data = _match_edge_attr_dim(graph_data, edge_dim)

    base_model = DielsAlderTransformer(input_dim=node_dim, edge_dim=edge_dim).to(device)
    if state_dict is not None:
        base_model.load_state_dict(state_dict)
    model = ConditionAwareTransformer(base_model).to(device)

    env = DielsAlderOptEnv(
        graph_data=graph_data,
        model=model,
        ts_mean=ts_mean,
        ts_std=ts_std,
        device=device,
        max_steps=1,
    )

    algo_key = ALGORITHM.lower().strip()
    if algo_key not in RL_ALGORITHMS:
        raise ValueError(f"Unknown algorithm {ALGORITHM!r}. Choose from {list(RL_ALGORITHMS.keys())}")
    AlgoClass = RL_ALGORITHMS[algo_key][0]
    zip_path = Path(RL_ZIP_PATH)
    if not zip_path.is_absolute():
        zip_path = _script_dir / zip_path
    rl_model = AlgoClass.load(str(zip_path), env=env)

    random_rewards = []
    for _ in range(N_RANDOM):
        obs, _ = env.reset()
        action = env.action_space.sample()
        _, r, _, _, _ = env.step(action)
        random_rewards.append(r)
    random_rewards = np.array(random_rewards)
    mean_r = float(np.mean(random_rewards))
    std_r = float(np.std(random_rewards, ddof=1)) if len(random_rewards) > 1 else 0.0

    obs, _ = env.reset()
    action, _ = rl_model.predict(obs, deterministic=True)
    _, rl_reward, _, _, _ = env.step(action)
    rl_reward = float(rl_reward)

    n_better = int((random_rewards < rl_reward).sum())
    stds_above = (rl_reward - mean_r) / std_r if std_r > 0 else float("nan")

    print(f"Random (n={N_RANDOM}): {mean_r:.4f} ± {std_r:.4f}")
    print(f"RL: {rl_reward:.4f}")
    print(f"RL beat {n_better}/{N_RANDOM} random trials.")
    print(f"RL is {stds_above:.2f} std above random mean.")


if __name__ == "__main__":
    main()
