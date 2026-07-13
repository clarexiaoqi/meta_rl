import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import torch
from scipy import stats


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FULL_SCRIPT = "ddpg_update.py"
ABLATION_SCRIPT = "ddpg_update_ablation.py"

FULL_RESULT_PATH = os.path.join(
    BASE_DIR,
    "ddpg",
    "ddpg_adapted.pt",
)

ABLATION_RESULT_PATH = os.path.join(
    BASE_DIR,
    "ddpg_ablation",
    "ddpg_adapted.pt",
)

SAVE_DIR = os.path.join(
    BASE_DIR,
    "ddpg_adaptation_statistics",
)

os.makedirs(SAVE_DIR, exist_ok=True)

SEEDS = list(range(10))


def run_script(script_name, seed):
    cmd = [
        "python3",
        os.path.join(BASE_DIR, script_name),
        "--seed",
        str(seed),
        "--no_plot",
    ]

    print("\n" + "=" * 70)
    print(f">>> Running {script_name} | seed = {seed}")
    print("=" * 70)

    subprocess.run(cmd, check=True)


def load_ddpg_result(result_path):
    ckpt = torch.load(
        result_path,
        map_location="cpu",
        weights_only=False,
    )

    if "returns" in ckpt:
        returns = np.array(ckpt["returns"], dtype=float)
    elif "episode_returns" in ckpt:
        returns = np.array(ckpt["episode_returns"], dtype=float)
    else:
        raise KeyError("Cannot find returns in ddpg_adapted.pt")

    best_returns_so_far = np.maximum.accumulate(returns)
    gaps = best_returns_so_far - returns

    result = {
        "num_iterations": len(returns),
        "initial_return": returns[0],
        "final_return": returns[-1],
        "best_return": np.max(returns),
        "mean_return": np.mean(returns),
        "std_return": np.std(returns, ddof=1),
        "worst_return": np.min(returns),
        "mean_gap_to_best": np.mean(gaps),
        "max_gap_to_best": np.max(gaps),
        "std_gap_to_best": np.std(gaps, ddof=1),
        "num_large_drops_gap_gt_1": int(np.sum(gaps > 1.0)),
        "num_large_drops_gap_gt_2": int(np.sum(gaps > 2.0)),
    }

    return result, returns, best_returns_so_far


def run_group(group_name, script_name, result_path):
    rows = []

    group_dir = os.path.join(SAVE_DIR, group_name)
    os.makedirs(group_dir, exist_ok=True)

    for seed in SEEDS:
        run_script(script_name, seed)

        metrics, returns, best_returns_so_far = load_ddpg_result(
            result_path
        )

        seed_dir = os.path.join(
            group_dir,
            f"seed_{seed}",
        )

        os.makedirs(seed_dir, exist_ok=True)

        shutil.copyfile(
            result_path,
            os.path.join(seed_dir, "ddpg_adapted.pt"),
        )

        pd.DataFrame(
            {
                "iteration": np.arange(1, len(returns) + 1),
                "current_return": returns,
                "best_return_so_far": best_returns_so_far,
                "gap_to_best": best_returns_so_far - returns,
            }
        ).to_csv(
            os.path.join(seed_dir, "adaptation_curve.csv"),
            index=False,
        )

        row = {
            "group": group_name,
            "seed": seed,
            **metrics,
        }

        rows.append(row)

        print("\n>>> Seed result")
        for k, v in row.items():
            print(f"{k}: {v}")

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(SAVE_DIR, f"{group_name}_raw_results.csv"),
        index=False,
    )

    return df


def summarize_group(df):
    metrics = [
        "initial_return",
        "final_return",
        "best_return",
        "mean_return",
        "std_return",
        "worst_return",
        "mean_gap_to_best",
        "max_gap_to_best",
        "std_gap_to_best",
        "num_large_drops_gap_gt_1",
        "num_large_drops_gap_gt_2",
    ]

    rows = []

    for metric in metrics:
        x = df[metric].values.astype(float)

        mean = np.mean(x)
        std = np.std(x, ddof=1)
        median = np.median(x)
        min_v = np.min(x)
        max_v = np.max(x)

        ci_low, ci_high = stats.t.interval(
            confidence=0.95,
            df=len(x) - 1,
            loc=mean,
            scale=std / np.sqrt(len(x)),
        )

        rows.append(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
                "median": median,
                "min": min_v,
                "max": max_v,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


def compare_groups(df_full, df_ablation):
    metrics = [
        "final_return",
        "best_return",
        "mean_return",
        "worst_return",
        "mean_gap_to_best",
        "max_gap_to_best",
        "std_gap_to_best",
        "num_large_drops_gap_gt_1",
        "num_large_drops_gap_gt_2",
    ]

    rows = []

    df_full = df_full.sort_values("seed")
    df_ablation = df_ablation.sort_values("seed")

    for metric in metrics:
        x = df_full[metric].values.astype(float)
        y = df_ablation[metric].values.astype(float)

        diff = x - y

        paired_t = stats.ttest_rel(x, y)

        try:
            wilcoxon = stats.wilcoxon(
                x,
                y,
                zero_method="wilcox",
                alternative="two-sided",
            )
            wilcoxon_stat = wilcoxon.statistic
            wilcoxon_p = wilcoxon.pvalue
        except ValueError:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan

        if np.std(diff, ddof=1) > 1e-12:
            cohen_dz = np.mean(diff) / np.std(diff, ddof=1)
        else:
            cohen_dz = np.nan

        rows.append(
            {
                "metric": metric,
                "full_mean": np.mean(x),
                "full_std": np.std(x, ddof=1),
                "ablation_mean": np.mean(y),
                "ablation_std": np.std(y, ddof=1),
                "mean_diff_full_minus_ablation": np.mean(diff),
                "paired_t_stat": paired_t.statistic,
                "paired_t_p_value": paired_t.pvalue,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p_value": wilcoxon_p,
                "cohen_dz": cohen_dz,
            }
        )

    return pd.DataFrame(rows)


def main():
    df_full = run_group(
        group_name="full_ddpg_adaptation",
        script_name=FULL_SCRIPT,
        result_path=FULL_RESULT_PATH,
    )

    df_ablation = run_group(
        group_name="vanilla_ddpg_adaptation",
        script_name=ABLATION_SCRIPT,
        result_path=ABLATION_RESULT_PATH,
    )

    df_all = pd.concat(
        [df_full, df_ablation],
        ignore_index=True,
    )

    df_all.to_csv(
        os.path.join(SAVE_DIR, "all_raw_results.csv"),
        index=False,
    )

    summary_full = summarize_group(df_full)
    summary_ablation = summarize_group(df_ablation)

    summary_full.to_csv(
        os.path.join(SAVE_DIR, "full_ddpg_adaptation_summary.csv"),
        index=False,
    )

    summary_ablation.to_csv(
        os.path.join(SAVE_DIR, "vanilla_ddpg_adaptation_summary.csv"),
        index=False,
    )

    comparison = compare_groups(
        df_full,
        df_ablation,
    )

    comparison.to_csv(
        os.path.join(SAVE_DIR, "ddpg_adaptation_statistical_comparison.csv"),
        index=False,
    )

    print("\n" + "=" * 80)
    print("FULL DDPG ADAPTATION SUMMARY")
    print("=" * 80)
    print(summary_full)

    print("\n" + "=" * 80)
    print("VANILLA DDPG ADAPTATION SUMMARY")
    print("=" * 80)
    print(summary_ablation)

    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON")
    print("=" * 80)
    print(comparison)

    print("\n>>> Results saved to:")
    print(SAVE_DIR)


if __name__ == "__main__":
    main()