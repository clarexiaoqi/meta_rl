from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Env_develop_mpc_draft import ContinuousBuildingControlEnvironment as BEnv
from mpc_controller import MPCController


def custom_plot(
    time,
    t_air,
    t_out,
    q_sg,
    sat_list,
    zat_list,
    lb_list,
    ub_list,
    save_dir,
    file_stem="mpc_control_profile",
    show_plot=True,
):
    """
    Create the same four-panel control profile used by offline_test.py.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    time = np.asarray(time, dtype=float)
    time_plot = time - time[0]

    fig = plt.figure(figsize=(10, 8.6))

    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.1, 1.1, 0.9],
        hspace=0.55,
        wspace=0.25,
    )

    ax_temp = fig.add_subplot(gs[0, :])
    ax_ctrl = fig.add_subplot(gs[1, :], sharex=ax_temp)
    ax_out = fig.add_subplot(gs[2, 0], sharex=ax_temp)
    ax_solar = fig.add_subplot(gs[2, 1], sharex=ax_temp)

    ticks = np.linspace(time_plot[0], time_plot[-1], 5)

    # Indoor temperature
    ax_temp.plot(
        time_plot,
        t_air,
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
    ax_ctrl.set_ylim(12.5, 26.5)
    ax_ctrl.legend(
        fontsize=10,
        loc="upper right",
    )
    ax_ctrl.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )

    # Outdoor temperature
    ax_out.plot(
        time_plot,
        t_out,
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

    # Solar gain
    ax_solar.plot(
        time_plot,
        q_sg,
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

    png_path = save_dir / f"{file_stem}.png"
    pdf_path = save_dir / f"{file_stem}.pdf"

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f">>> Saved plot to {png_path}")
    print(f">>> Saved plot to {pdf_path}")


def run_mpc_test(
    start=17664.0,
    end=19872.5,
    horizon=6,
    maxiter=100,
    ftol=1e-6,
    sat_smooth_weight=0.0,
    zat_smooth_weight=0.0,
    default_sat=14.5,
    default_zat=23.0,
    data_file="weather_data_2013_to_2017_summer_pandas.csv",
    show_plot=True,
):
    """
    Run a complete closed-loop MPC simulation on the same interval as offline_test.py.

    Each outer step:
      1. optimize a horizon-long SAT/ZAT sequence,
      2. execute only the first action in the real environment,
      3. repeat from the new state.
    """
    env = BEnv(
        data_file=data_file,
        dt=1800.0,
        start=float(start),
        end=float(end),
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369,
        lb_set=22.0,
        ub_set=24.0,
    )

    obs = env.reset()

    controller = MPCController(
        env=env,
        horizon=int(horizon),
        method="SLSQP",
        maxiter=int(maxiter),
        ftol=float(ftol),
        sat_smooth_weight=float(sat_smooth_weight),
        zat_smooth_weight=float(zat_smooth_weight),
        default_action=[default_sat, default_zat],
    )

    controller.reset()

    # Time-series records
    time_list: List[float] = []
    t_air_list: List[float] = []
    t_env_list: List[float] = []
    t_out_list: List[float] = []
    q_sg_list: List[float] = []
    q_int_list: List[float] = []

    sat_list: List[float] = []
    zat_list: List[float] = []
    lb_list: List[float] = []
    ub_list: List[float] = []

    reward_list: List[float] = []
    energy_step_list: List[float] = []
    energy_cumulative_list: List[float] = []
    exceed_step_list: List[float] = []
    exceed_cumulative_list: List[float] = []
    violation_cumulative_list: List[float] = []

    solver_success_list: List[bool] = []
    solver_fallback_list: List[bool] = []
    solver_time_list: List[float] = []
    solver_iterations_list: List[int] = []
    solver_nfev_list: List[int] = []
    solver_objective_list: List[float] = []
    solver_message_list: List[str] = []

    cumulative_energy = 0.0
    cumulative_exceedance = 0.0
    cumulative_violation_hours = 0.0

    dt_hour = env.dt / 3600.0
    total_steps_expected = int(np.ceil((end - start) / dt_hour))

    step_idx = 0
    done = False

    while not done:
        step_idx += 1

        action, diagnostics = controller.solve()

        obs, reward, done, info = env.step(action)
        controller.set_previous_action(action)

        cumulative_energy += info["TotalEnergy_kWh"]

        exceed_step_degC_hr = (
            info["TempExceed_degC"] * dt_hour
        )
        cumulative_exceedance += exceed_step_degC_hr

        if info["TempExceed_degC"] > 0.0:
            cumulative_violation_hours += dt_hour

        # Record post-action state at env.t.
        time_list.append(float(env.t))
        t_air_list.append(float(info["T_zone_raw"]))
        t_env_list.append(float(info["T_env_raw"]))
        t_out_list.append(float(info["T_out_raw"]))
        q_sg_list.append(float(info["Qsg_raw"]))
        q_int_list.append(float(info["Qint_raw"]))

        sat_list.append(float(info["SAT_sp"]))
        zat_list.append(float(info["ZAT_sp_used"]))
        lb_list.append(float(info["lb"]))
        ub_list.append(float(info["ub"]))

        reward_list.append(float(reward))
        energy_step_list.append(float(info["TotalEnergy_kWh"]))
        energy_cumulative_list.append(float(cumulative_energy))
        exceed_step_list.append(float(exceed_step_degC_hr))
        exceed_cumulative_list.append(float(cumulative_exceedance))
        violation_cumulative_list.append(
            float(cumulative_violation_hours)
        )

        solver_success_list.append(bool(diagnostics.success))
        solver_fallback_list.append(bool(diagnostics.used_fallback))
        solver_time_list.append(float(diagnostics.solve_time_sec))
        solver_iterations_list.append(int(diagnostics.iterations))
        solver_nfev_list.append(
            int(diagnostics.function_evaluations)
        )
        solver_objective_list.append(float(diagnostics.objective))
        solver_message_list.append(str(diagnostics.message))

        if (
            step_idx == 1
            or step_idx % 10 == 0
            or done
        ):
            print(
                f"[MPC step {step_idx:04d}/{total_steps_expected:04d}] "
                f"t={env.t:.1f} h | "
                f"SAT={action[0]:.3f} | "
                f"ZAT={action[1]:.3f} | "
                f"Tzone={info['T_zone_raw']:.3f} °C | "
                f"E={cumulative_energy:.4f} kWh | "
                f"success={diagnostics.success} | "
                f"solve={diagnostics.solve_time_sec:.3f} s"
            )

    env.close()

    total_return = float(np.sum(reward_list))
    success_rate = (
        float(np.mean(solver_success_list))
        if solver_success_list
        else float("nan")
    )
    fallback_rate = (
        float(np.mean(solver_fallback_list))
        if solver_fallback_list
        else float("nan")
    )
    avg_solve_time = (
        float(np.mean(solver_time_list))
        if solver_time_list
        else float("nan")
    )
    max_solve_time = (
        float(np.max(solver_time_list))
        if solver_time_list
        else float("nan")
    )
    avg_iterations = (
        float(np.mean(solver_iterations_list))
        if solver_iterations_list
        else float("nan")
    )
    avg_nfev = (
        float(np.mean(solver_nfev_list))
        if solver_nfev_list
        else float("nan")
    )

    summary: Dict[str, float] = {
        "method": "Oracle MPC",
        "start_hour": float(start),
        "end_hour": float(end),
        "duration_hours": float(end - start),
        "horizon_steps": int(horizon),
        "horizon_hours": float(horizon * dt_hour),
        "total_steps": int(len(reward_list)),
        "total_return": total_return,
        "energy_kwh": float(cumulative_energy),
        "violation_hours": float(cumulative_violation_hours),
        "temp_exceedance_degC_hr": float(cumulative_exceedance),
        "solver_success_rate": success_rate,
        "solver_fallback_rate": fallback_rate,
        "avg_solver_time_sec": avg_solve_time,
        "max_solver_time_sec": max_solve_time,
        "avg_solver_iterations": avg_iterations,
        "avg_solver_function_evaluations": avg_nfev,
    }

    profile_df = pd.DataFrame(
        {
            "time_hour": time_list,
            "Tenv_mpc": t_env_list,
            "Tair_mpc": t_air_list,
            "Tout": t_out_list,
            "Qsg": q_sg_list,
            "Qint": q_int_list,
            "sat_mpc": sat_list,
            "zat_mpc": zat_list,
            "lb": lb_list,
            "ub": ub_list,
            "reward_mpc": reward_list,
            "energy_step_mpc_kwh": energy_step_list,
            "energy_cumulative_mpc_kwh": energy_cumulative_list,
            "temp_exceed_step_mpc_degC_hr": exceed_step_list,
            "temp_exceed_cumulative_mpc_degC_hr": exceed_cumulative_list,
            "violation_hours_cumulative_mpc": violation_cumulative_list,
            "solver_success": solver_success_list,
            "solver_used_fallback": solver_fallback_list,
            "solver_time_sec": solver_time_list,
            "solver_iterations": solver_iterations_list,
            "solver_function_evaluations": solver_nfev_list,
            "solver_objective": solver_objective_list,
            "solver_message": solver_message_list,
        }
    )

    summary_df = pd.DataFrame([summary])

    results_dir = Path("results") / "mpc"
    plots_dir = Path("plots") / "mpc"

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    profile_path = results_dir / "mpc_results_profile_full_interval.csv"
    summary_path = results_dir / "mpc_results_summary_full_interval.csv"

    profile_df.to_csv(profile_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Match offline_test.py exactly:
    # start_idx = 2100
    # window = 400 steps = 200 hours
    plot_start_idx = 2100
    plot_window = 400
    plot_end_idx = min(plot_start_idx + plot_window, len(time_list))

    if plot_start_idx >= len(time_list):
        raise ValueError(
            f"Plot start index {plot_start_idx} exceeds available "
            f"trajectory length {len(time_list)}."
        )

    custom_plot(
        time=np.asarray(time_list)[plot_start_idx:plot_end_idx],
        t_air=np.asarray(t_air_list)[plot_start_idx:plot_end_idx],
        t_out=np.asarray(t_out_list)[plot_start_idx:plot_end_idx],
        q_sg=np.asarray(q_sg_list)[plot_start_idx:plot_end_idx],
        sat_list=np.asarray(sat_list)[plot_start_idx:plot_end_idx],
        zat_list=np.asarray(zat_list)[plot_start_idx:plot_end_idx],
        lb_list=np.asarray(lb_list)[plot_start_idx:plot_end_idx],
        ub_list=np.asarray(ub_list)[plot_start_idx:plot_end_idx],
        save_dir=plots_dir,
        file_stem="mpc_control_profile_full_interval",
        show_plot=show_plot,
    )

    print("\n" + "=" * 64)
    print("MPC TEST SUMMARY")
    print("=" * 64)
    print(f"Simulation interval: {start:.1f} to {end:.1f} h")
    print(
        f"Prediction horizon: {horizon} steps "
        f"({horizon * dt_hour:.1f} h)"
    )
    print(f"Total Return: {total_return:.8f}")
    print(f"Energy Use in kWh: {cumulative_energy:.6f}")
    print(
        "# of Hours out of Bounds: "
        f"{cumulative_violation_hours:.4f}"
    )
    print(
        "Temperature Exceedance in degC-hr: "
        f"{cumulative_exceedance:.6f}"
    )
    print(f"Solver Success Rate: {100.0 * success_rate:.2f}%")
    print(f"Fallback Rate: {100.0 * fallback_rate:.2f}%")
    print(f"Average Solver Time: {avg_solve_time:.6f} s")
    print(f"Maximum Solver Time: {max_solve_time:.6f} s")
    print(f"Average Solver Iterations: {avg_iterations:.3f}")
    print(f"Average Function Evaluations: {avg_nfev:.3f}")
    print(f">>> Saved profile to {profile_path}")
    print(f">>> Saved summary to {summary_path}")
    print("=" * 64)

    return summary, profile_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run closed-loop Oracle MPC test on the RL offline-test interval."
    )

    parser.add_argument("--start", type=float, default=17664.0)
    parser.add_argument("--end", type=float, default=19872.5)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--ftol", type=float, default=1e-6)

    parser.add_argument(
        "--sat_smooth_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--zat_smooth_weight",
        type=float,
        default=0.0,
    )

    parser.add_argument("--default_sat", type=float, default=14.5)
    parser.add_argument("--default_zat", type=float, default=23.0)

    parser.add_argument(
        "--data_file",
        type=str,
        default="weather_data_2013_to_2017_summer_pandas.csv",
    )

    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Save plots without opening a window.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_mpc_test(
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        maxiter=args.maxiter,
        ftol=args.ftol,
        sat_smooth_weight=args.sat_smooth_weight,
        zat_smooth_weight=args.zat_smooth_weight,
        default_sat=args.default_sat,
        default_zat=args.default_zat,
        data_file=args.data_file,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()