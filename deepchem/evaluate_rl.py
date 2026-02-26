import os
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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

# --- Config ---
# The script will look for: trainedmodels/default_ppo_row0.zip, etc.
MODEL_DIR = _script_dir / "trainedmodels"
ALGOS_TO_EVAL = ["ppo", "a2c", "sac", "td3"]
DATA_ROOT = _script_dir
MODEL_PATH = None
STATS_PATH = None
DATA_FILE = DEFAULT_DATA_FILE
ROW_IDX = 0
N_RANDOM = 30

def main():
    # 1. Environment Setup (Common to all evaluations)
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
    
    # Infer dimensions from the base chemistry model
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

    # 2. Baseline: Random Actions
    print(f"Evaluating Random Baseline ({N_RANDOM} trials)...")
    random_rewards = []
    for _ in range(N_RANDOM):
        env.reset()
        action = env.action_space.sample()
        _, r, _, _, _ = env.step(action)
        random_rewards.append(r)
    
    random_rewards = np.array(random_rewards)
    mean_r = float(np.mean(random_rewards))
    std_r = float(np.std(random_rewards, ddof=1)) if len(random_rewards) > 1 else 0.0

    # 3. Evaluate each RL Algorithm
    results = {"Random": mean_r}
    
    for algo_key in ALGOS_TO_EVAL:
        zip_name = f"default_{algo_key.lower()}_row{ROW_IDX}.zip"
        zip_path = MODEL_DIR / zip_name
        
        if not zip_path.exists():
            print(f"Skipping {algo_key}: File {zip_path} not found.")
            continue

        print(f"Evaluating {algo_key.upper()}...")
        try:
            AlgoClass = RL_ALGORITHMS[algo_key.lower()][0]
            rl_model = AlgoClass.load(str(zip_path), env=env)

            obs, _ = env.reset()
            action, _ = rl_model.predict(obs, deterministic=True)
            _, rl_reward, _, _, _ = env.step(action)
            
            results[algo_key.upper()] = float(rl_reward)
        except Exception as e:
            print(f"Could not evaluate {algo_key}: {e}")

    # 4. Generate Graph
    create_comparison_plot(results, mean_r, std_r)

def create_comparison_plot(results, mean_r, std_r):
    names = list(results.keys())
    values = list(results.values())
    
    plt.figure(figsize=(10, 6))
    
    # Define colors: Gray for random, Green for beating random, Red for failing
    colors = []
    for name in names:
        if name == "Random":
            colors.append('gray')
        else:
            colors.append('#2ca02c' if results[name] >= mean_r else '#d62728')

    bars = plt.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add Error Bar for Random Baseline
    plt.errorbar(0, mean_r, yerr=std_r, fmt='none', ecolor='black', capsize=10, label='Random Std Dev')
    
    # Baseline horizontal line
    plt.axhline(y=mean_r, color='black', linestyle='--', alpha=0.5)
    
    plt.title(f"RL Algorithm Performance vs. Random Baseline (Row {ROW_IDX})", fontsize=14)
    plt.ylabel("Reward (Standardized Yield)", fontsize=12)
    plt.xlabel("Algorithm", fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    # Annotate bars with exact values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), 
                 f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = "algorithm_comparison.png"
    plt.savefig(output_path)
    print(f"\nEvaluation complete. Graph saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()