import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import torch
from scipy import stats

from Env_develop import ContinuousBuildingControlEnvironment as BEnv
from train_ppo_cps import ActorCritic


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FULL_SCRIPT = "meta-rl.py"
ABLATION_SCRIPT = "meta_rl_ablation.py"

MODEL_PATH = os.path.join(BASE_DIR, "model", "final_actor.pth")

SAVE_DIR = os.path.join(BASE_DIR, "statistical_results")
os.makedirs(SAVE_DIR, exist_ok=True)

SEEDS = list(range(10))


def run_training(script_name, seed):
    cmd = [
        "python3",
        os.path.join(BASE_DIR, script_name),
        "--seed",
        str(seed),
    ]

    print("\n" + "=" * 70)
    print(f">>> Running {script_name} with seed = {seed}")
    print("=" * 70)

    subprocess.run(cmd, check=True)


def evaluate_final_actor():
    env = BEnv(
        data_file="weather_data_2013_to_2017_summer_pandas.csv",
        dt=1800.0,
        start=17664,
        end=19872.5,
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369,
        lb_set=22.0,
        ub_set=24.0,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    model = ActorCritic(
        obs_dim,
        act_dim,
        act_low,
        act_high,
    )

    actor_state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.actor.load_state_dict(actor_state)
    model.eval()

    obs = env.reset()
    done = False

    total_return = 0.0
    total_energy = 0.0
    violation_hours = 0.0
    temp_exceedance = 0.0

    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
            ).unsqueeze(0)

            action, _, _ = model.get_action_and_value(obs_t)
            action = action.squeeze(0).cpu().numpy()

            obs, reward, done, info = env.step(action)

            total_return += reward
            total_energy += info["TotalEnergy_kWh"]

            if info["TempExceed_degC"] > 0:
                violation_hours += env.dt / 3600.0

            temp_exceedance += info["TempExceed_degC"] * env.dt / 3600.0

    return {
        "return": total_return,
        "energy_kwh": total_energy,
        "violation_hours": violation_hours,
        "temp_exceedance_degC_hr": temp_exceedance,
    }


def run_group(group_name, script_name):
    rows = []

    group_dir = os.path.join(SAVE_DIR, group_name)
    os.makedirs(group_dir, exist_ok=True)

    for seed in SEEDS:
        run_training(script_name, seed)

        metrics = evaluate_final_actor()

        saved_model_path = os.path.join(
            group_dir,
            f"final_actor_seed_{seed}.pth",
        )

        shutil.copyfile(
            MODEL_PATH,
            saved_model_path,
        )

        row = {
            "group": group_name,
            "seed": seed,
            **metrics,
        }

        rows.append(row)

        print("\n>>> Evaluation result")
        for k, v in row.items():
            print(f"{k}: {v}")

    df = pd.DataFrame(rows)

    csv_path = os.path.join(
        SAVE_DIR,
        f"{group_name}_raw_results.csv",
    )

    df.to_csv(csv_path, index=False)

    return df


def summarize_group(df):
    metrics = [
        "return",
        "energy_kwh",
        "violation_hours",
        "temp_exceedance_degC_hr",
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
        "return",
        "energy_kwh",
        "violation_hours",
        "temp_exceedance_degC_hr",
    ]

    rows = []

    for metric in metrics:
        x = df_full.sort_values("seed")[metric].values.astype(float)
        y = df_ablation.sort_values("seed")[metric].values.astype(float)

        diff = x - y

        paired_t = stats.ttest_rel(x, y)
        wilcoxon = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")

        cohen_dz = np.mean(diff) / np.std(diff, ddof=1)

        rows.append(
            {
                "metric": metric,
                "full_mean": np.mean(x),
                "full_std": np.std(x, ddof=1),
                "ablation_mean": np.mean(y),
                "ablation_std": np.std(y, ddof=1),
                "mean_difference_full_minus_ablation": np.mean(diff),
                "paired_t_stat": paired_t.statistic,
                "paired_t_p_value": paired_t.pvalue,
                "wilcoxon_stat": wilcoxon.statistic,
                "wilcoxon_p_value": wilcoxon.pvalue,
                "cohen_dz": cohen_dz,
            }
        )

    return pd.DataFrame(rows)


def main():
    df_full = run_group(
        group_name="full_method",
        script_name=FULL_SCRIPT,
    )

    df_ablation = run_group(
        group_name="vanilla_ablation",
        script_name=ABLATION_SCRIPT,
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
        os.path.join(SAVE_DIR, "full_method_summary.csv"),
        index=False,
    )

    summary_ablation.to_csv(
        os.path.join(SAVE_DIR, "vanilla_ablation_summary.csv"),
        index=False,
    )

    comparison = compare_groups(
        df_full,
        df_ablation,
    )

    comparison.to_csv(
        os.path.join(SAVE_DIR, "statistical_comparison.csv"),
        index=False,
    )

    print("\n" + "=" * 80)
    print("FULL METHOD SUMMARY")
    print("=" * 80)
    print(summary_full)

    print("\n" + "=" * 80)
    print("VANILLA ABLATION SUMMARY")
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