import os
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def evaluate_saved_models_for_env(
    env,
    rl_algorithms,
    model_dir: Path,
    model_prefix: str,
    row_idx: int,
    algo_keys,
    n_random: int = 30,
):
    """Evaluate saved RL models against random baseline for one reaction row."""
    random_rewards = []
    for _ in range(n_random):
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        random_rewards.append(reward)

    random_rewards = np.array(random_rewards, dtype=float)
    mean_r = float(np.mean(random_rewards))
    std_r = float(np.std(random_rewards, ddof=1)) if len(random_rewards) > 1 else 0.0

    results = {"Random": mean_r}
    details = []

    for algo_key in algo_keys:
        algo_key = algo_key.lower().strip()
        zip_name = f"{model_prefix}_{algo_key}_row{row_idx}.zip"
        zip_path = model_dir / zip_name

        if not zip_path.exists():
            details.append(f"Skipping {algo_key.upper()}: file not found at {zip_path}")
            continue

        try:
            algo_class = rl_algorithms[algo_key][0]
            rl_model = algo_class.load(str(zip_path), env=env)

            obs, _ = env.reset()
            action, _ = rl_model.predict(obs, deterministic=True)
            _, rl_reward, _, _, rl_info = env.step(action)
            rl_reward = float(rl_reward)
            results[algo_key.upper()] = rl_reward

            n_better = int((random_rewards < rl_reward).sum())
            stds_above = (rl_reward - mean_r) / std_r if std_r > 0 else float("nan")
            details.extend(
                [
                    f"{algo_key.upper()} reward: {rl_reward:.4f}",
                    f"{algo_key.upper()} beat {n_better}/{n_random} random trials.",
                    f"{algo_key.upper()} std above random mean: {stds_above:.2f}",
                    f"{algo_key.upper()} action (normalized 0-1): {np.array(action, dtype=float)}",
                    (
                        f"{algo_key.upper()} conditions: "
                        f"T={rl_info['temp_k']:.1f} K, "
                        f"[diene]={rl_info['conc_diene_m']:.3f} M, "
                        f"[dienophile]={rl_info['conc_dienophile_m']:.3f} M, "
                        f"t={rl_info['t_sec']:.1f} s, "
                        f"Lewis acid={rl_info['lewis_equiv']:.2f} eq, "
                        f"solvent_idx={rl_info['solvent_idx']}"
                    ),
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive path
            details.append(f"Could not evaluate {algo_key.upper()}: {exc}")

    return {
        "results": results,
        "random_mean": mean_r,
        "random_std": std_r,
        "details": details,
    }


def create_comparison_plot(results, mean_r, std_r, row_idx, output_path: Path, show: bool = False):
    names = list(results.keys())
    values = list(results.values())
    plt.figure(figsize=(10, 6))

    colors = []
    for name in names:
        if name == "Random":
            colors.append("gray")
        else:
            colors.append("#2ca02c" if results[name] >= mean_r else "#d62728")

    bars = plt.bar(names, values, color=colors, alpha=0.8, edgecolor="black")
    plt.errorbar(0, mean_r, yerr=std_r, fmt="none", ecolor="black", capsize=10, label="Random Std Dev")
    plt.axhline(y=mean_r, color="black", linestyle="--", alpha=0.5)

    plt.title(f"RL Algorithm Performance vs. Random Baseline (Row {row_idx})", fontsize=14)
    plt.ylabel("Reward (Standardized Yield)", fontsize=12)
    plt.xlabel("Algorithm", fontsize=12)
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    max_val = max(values) if values else 0.0
    y_offset = max(max_val * 0.01, 1e-6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + y_offset,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def build_evaluation_text(row_idx: int, data_path: Path, model_dir: Path, eval_result: dict):
    lines = [
        f"Row index: {row_idx}",
        f"Data file: {data_path}",
        f"Model directory: {model_dir}",
        f"Random baseline mean: {eval_result['random_mean']:.6f}",
        f"Random baseline std: {eval_result['random_std']:.6f}",
        "",
    ]
    lines.extend(eval_result["details"])
    return "\n".join(lines) + "\n"


def _load_optimizer_deps():
    from reaction_optimizer import (
        DEFAULT_DATA_FILE,
        ConditionAwareTransformer,
        DielsAlderOptEnv,
        DielsAlderTransformer,
        RL_ALGORITHMS,
        _infer_edge_dim_from_state,
        _infer_node_dim_from_state,
        _match_edge_attr_dim,
        _match_node_attr_dim,
        _resolve_run_paths,
        build_graph_from_row,
        load_stats,
    )

    return {
        "DEFAULT_DATA_FILE": DEFAULT_DATA_FILE,
        "ConditionAwareTransformer": ConditionAwareTransformer,
        "DielsAlderOptEnv": DielsAlderOptEnv,
        "DielsAlderTransformer": DielsAlderTransformer,
        "RL_ALGORITHMS": RL_ALGORITHMS,
        "_infer_edge_dim_from_state": _infer_edge_dim_from_state,
        "_infer_node_dim_from_state": _infer_node_dim_from_state,
        "_match_edge_attr_dim": _match_edge_attr_dim,
        "_match_node_attr_dim": _match_node_attr_dim,
        "_resolve_run_paths": _resolve_run_paths,
        "build_graph_from_row": build_graph_from_row,
        "load_stats": load_stats,
    }


# --- Standalone script defaults ---
MODEL_DIR = _script_dir / "trainedmodels"
ALGOS_TO_EVAL = ["ppo", "a2c", "sac", "td3"]
DATA_ROOT = _script_dir
MODEL_PATH = None
STATS_PATH = None
ROW_IDX = 0
N_RANDOM = 30


def main():
    deps = _load_optimizer_deps()
    data_file = deps["DEFAULT_DATA_FILE"]
    data_root = DATA_ROOT
    model_path, stats_path = deps["_resolve_run_paths"](data_root, MODEL_PATH, STATS_PATH)
    stats = deps["load_stats"](data_root, stats_path)
    ts_mean = float(stats["ts_mean"])
    ts_std = float(stats["ts_std"])
    feature_stats = stats["feature_stats"]

    data_path = data_root / data_file
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
        expected_edge_dim = deps["_infer_edge_dim_from_state"](state_dict)
        expected_node_dim = deps["_infer_node_dim_from_state"](state_dict)

    graph_data = deps["build_graph_from_row"](row, feature_stats, expected_node_dim, expected_edge_dim)
    graph_data = graph_data.to(device)
    edge_dim = graph_data.edge_attr.size(1)
    node_dim = graph_data.x.size(1)
    graph_data = deps["_match_node_attr_dim"](graph_data, node_dim)
    graph_data = deps["_match_edge_attr_dim"](graph_data, edge_dim)

    base_model = deps["DielsAlderTransformer"](input_dim=node_dim, edge_dim=edge_dim).to(device)
    if state_dict is not None:
        base_model.load_state_dict(state_dict)
    model = deps["ConditionAwareTransformer"](base_model).to(device)

    env = deps["DielsAlderOptEnv"](
        graph_data=graph_data,
        model=model,
        ts_mean=ts_mean,
        ts_std=ts_std,
        device=device,
        max_steps=1,
    )

    eval_result = evaluate_saved_models_for_env(
        env=env,
        rl_algorithms=deps["RL_ALGORITHMS"],
        model_dir=MODEL_DIR,
        model_prefix="default",
        row_idx=ROW_IDX,
        algo_keys=ALGOS_TO_EVAL,
        n_random=N_RANDOM,
    )
    output_path = _script_dir / "algorithm_comparison.png"
    create_comparison_plot(
        eval_result["results"],
        eval_result["random_mean"],
        eval_result["random_std"],
        ROW_IDX,
        output_path,
        show=True,
    )
    print(build_evaluation_text(ROW_IDX, data_path, MODEL_DIR, eval_result))
    print(f"Evaluation complete. Graph saved to {output_path}")


if __name__ == "__main__":
    main()