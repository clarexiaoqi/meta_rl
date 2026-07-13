import numpy as np
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from Env_develop import ContinuousBuildingControlEnvironment as BEnv
from ddpg_torch import Actor


# =========================
# CUSTOM PLOT
# =========================
def custom_plot(
    T_air, time, T_out, Q_SG,
    sat_list, zat_list, energy_list, penalty_list,
    lb_list, ub_list, idx, folder_name
):

    save_dir = os.path.join("plots", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    time_plot = time - time[0]

    fig = plt.figure(figsize=(10, 8.6))

    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.1, 1.1, 0.9],
        hspace=0.55,
        wspace=0.25,
    )

    ax_temp = fig.add_subplot(gs[0, :])
    ax_ctrl = fig.add_subplot(gs[1, :], sharex=ax_temp)
    ax_out = fig.add_subplot(gs[2, 0], sharex=ax_temp)
    ax_solar = fig.add_subplot(gs[2, 1], sharex=ax_temp)

    ticks = np.linspace(time_plot[0], time_plot[-1], 5)

    # Indoor air temperature
    ax_temp.plot(
        time_plot,
        T_air,
        linewidth=2.2,
        label="Indoor Air Temperature",
    )

    ax_temp.plot(
        time_plot,
        lb_list,
        "--",
        linewidth=1.1,
        alpha=0.45,
        label="Lower Bound",
    )

    ax_temp.plot(
        time_plot,
        ub_list,
        "--",
        linewidth=1.1,
        alpha=0.45,
        label="Upper Bound",
    )

    ax_temp.set_title(
        "Indoor Air Temperature",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    ax_temp.set_ylabel("Temperature (°C)", fontsize=12)

    ax_temp.legend(
        fontsize=7.5,
        loc="lower right",
        framealpha=0.65,
        frameon=True,
    )

    ax_temp.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )

    # Control inputs
    ax_ctrl.plot(
        time_plot,
        sat_list,
        linewidth=1.8,
        label="SAT",
    )

    ax_ctrl.plot(
        time_plot,
        zat_list,
        linewidth=1.8,
        label="ZAT",
    )

    ax_ctrl.set_title(
        "Control Inputs (SAT and ZAT)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    ax_ctrl.set_ylabel("Setpoint (°C)", fontsize=12)
    ax_ctrl.set_ylim(12.5, 24.2)

    ax_ctrl.legend(
        fontsize=10,
        loc="upper right",
    )

    ax_ctrl.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )

    # Outdoor air temperature
    ax_out.plot(
        time_plot,
        T_out,
        linewidth=1.6,
        alpha=0.85,
    )

    ax_out.set_title(
        "Outdoor Air Temperature",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )

    ax_out.set_ylabel("Temperature (°C)", fontsize=12)
    ax_out.set_xlabel("Time (h)", fontsize=12)
    ax_out.set_xticks(ticks)

    ax_out.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )

    # Solar heat gain
    ax_solar.plot(
        time_plot,
        Q_SG,
        linewidth=1.6,
        alpha=0.85,
    )

    ax_solar.set_title(
        "Solar Heat Gain",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )

    ax_solar.set_ylabel("Heat Gain (W)", fontsize=12)
    ax_solar.set_xlabel("Time (h)", fontsize=12)
    ax_solar.set_xticks(ticks)

    ax_solar.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )

    for ax in [ax_temp, ax_ctrl, ax_out, ax_solar]:
        ax.tick_params(axis="both", labelsize=10)

    ax_temp.tick_params(labelbottom=True)
    ax_ctrl.tick_params(labelbottom=True)

    plt.tight_layout()

    png_path = os.path.join(save_dir, f"{idx}_control_profile.png")
    pdf_path = os.path.join(save_dir, f"{idx}_control_profile.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.show()

    print(f">>> Saved plot to {png_path}")
    print(f">>> Saved plot to {pdf_path}")

# =========================
# TEST POLICY
# =========================
def test_policy(
    policy_file,
    actor_name="best_actor",
    start=17664.,
    end=19872.5,
    data_file="weather_data_2013_to_2017_summer_pandas.csv"
):

    env = BEnv(
        data_file=data_file,
        start=start,
        end=end,
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369,
        lb_set=22.,
        ub_set=24.,
    )

    obs = env.reset()
    done = False

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    actor = Actor(
        obs_dim,
        act_dim,
        act_low,
        act_high,
    )

    checkpoint = torch.load(
        policy_file,
        map_location="cpu",
        weights_only=False,
    )

    if actor_name not in checkpoint:
        raise KeyError(
            f"{actor_name} not found in checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    actor.load_state_dict(checkpoint[actor_name])
    actor.eval()

    print(f">>> Loaded actor: {actor_name}")
    print(f">>> Action low: {act_low}")
    print(f">>> Action high: {act_high}")

    obs_list = []
    reward_list = []

    sat_list = []
    zat_list = []

    raw_action_list = []

    energy_list = [0.0]
    penalty_list = [0.0]
    temp_metric_list = [0.0]

    lb_list = []
    ub_list = []

    with torch.no_grad():
        while True:

            if not done:
                obs_list.append(obs.copy())

                obs_t = torch.tensor(
                    obs,
                    dtype=torch.float32,
                ).reshape(1, -1)

                action_val = actor(obs_t)
                action_val = action_val.cpu().numpy()[0]

                raw_action_list.append(action_val.copy())

                obs, reward, done, dic = env.step(action_val)

                reward_list.append(reward)

                sat_list.append(dic["SAT_sp"])
                zat_list.append(dic["ZAT_sp_used"])

                energy_list.append(
                    energy_list[-1] + dic["TotalEnergy_kWh"]
                )

                penalty_step = 0.5 if dic["TempExceed_degC"] > 0 else 0.0

                penalty_list.append(
                    penalty_list[-1] + penalty_step
                )

                temp_metric_list.append(
                    temp_metric_list[-1]
                    + dic["TempExceed_degC"] * 0.5
                )

                # Important:
                # Use dynamic day/night bounds from environment info.
                lb_list.append(dic["lb"])
                ub_list.append(dic["ub"])

            if done:
                break

    env.close()

    low = env.low
    high = env.high

    obs_arr = np.array(obs_list)

    T_air = obs_arr[:, 1] * (high[1] - low[1]) + low[1]
    time = np.linspace(start, end, len(T_air))
    T_out = obs_arr[:, 3] * (high[3] - low[3]) + low[3]
    Q_SG = obs_arr[:, 4] * (high[4] - low[4]) + low[4]

    raw_action_arr = np.array(raw_action_list)

    print(">>> Raw action min:", raw_action_arr.min(axis=0))
    print(">>> Raw action max:", raw_action_arr.max(axis=0))
    print(">>> Raw action mean:", raw_action_arr.mean(axis=0))

    return (
        T_air,
        time,
        T_out,
        Q_SG,
        np.array(sat_list),
        np.array(zat_list),
        np.array(energy_list[1:]),
        np.array(penalty_list[1:]),
        np.array(temp_metric_list[1:]),
        np.array(lb_list),
        np.array(ub_list),
        np.array(reward_list),
        np.array(raw_action_list),
    )


# =========================
# MAIN
# =========================
def main():

    torch.manual_seed(0)
    np.random.seed(0)

    data_file = "weather_data_2013_to_2017_summer_pandas.csv"

    policy_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ddpg",
        "ddpg_adapted.pt"
    )

    actor_name = "best_actor"
    # actor_name = "final_actor"
    # actor_name = "last_best_actor"
    # actor_name = "last_min_exceed_actor"

    for idx in range(0, 1):

        (
            T_air,
            time,
            T_out,
            Q_SG,
            sat_list,
            zat_list,
            energy_list,
            penalty_list,
            temp_metric_list,
            lb_list,
            ub_list,
            reward_list,
            raw_action_list,
        ) = test_policy(
            start=17664.,
            end=19872.5,
            data_file=data_file,
            policy_file=policy_file,
            actor_name=actor_name,
        )

        # =========================
        # Plot window
        # =========================
        start_idx = 2100
        window = 400  # 400 steps = 200 hours ≈ 8.3 days
        end_idx = start_idx + window

        custom_plot(
            T_air[start_idx:end_idx],
            time[start_idx:end_idx],
            T_out[start_idx:end_idx],
            Q_SG[start_idx:end_idx],
            sat_list[start_idx:end_idx],
            zat_list[start_idx:end_idx],
            energy_list[start_idx:end_idx],
            penalty_list[start_idx:end_idx],
            lb_list[start_idx:end_idx],
            ub_list[start_idx:end_idx],
            idx,
            "offline",
        )

        print(idx)

    d = {
        "energy_true": energy_list,
        "penalty_true": penalty_list,
        "exceedance_true": temp_metric_list,
        "reward_true": reward_list,
    }

    pd.DataFrame(d).to_csv(
        "results_true.csv",
        index=False,
    )

    d_profile = {
        "lb": lb_list,
        "ub": ub_list,
        "Qsg": Q_SG,
        "Tout": T_out,
        "Tair_true": T_air,
        "sat_true": sat_list,
        "zat_true": zat_list,
        "raw_action_sat": raw_action_list[:, 0],
        "raw_action_zat": raw_action_list[:, 1],
    }

    pd.DataFrame(d_profile).to_csv(
        "results_profile.csv",
        index=False,
    )

    print("Tested actor:", actor_name)
    print("Energy Use in kWh: %.2f" % energy_list[-1])
    print("# of Hours out of Bounds: %.2f" % penalty_list[-1])
    print("Temperature Exceedance in degC-hr: %.2f" % temp_metric_list[-1])


if __name__ == "__main__":
    main()



