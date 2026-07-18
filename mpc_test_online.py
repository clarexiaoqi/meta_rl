from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Env_develop_mpc_draft import ContinuousBuildingControlEnvironment as BEnv
from mpc_controller import MPCController
from mpc_model import (
    OnlineParameterLearningMPC,
    PARAMETER_NAMES,
    mpc_state_from_env,
)


# =============================================================================
# Parameter settings
# =============================================================================

TRUE_PARAMETERS: Dict[str, float] = {
    "C_env": 3.1996e6,
    "C_air": 3.5187e5,
    "R_rc": 0.00706,
    "R_oe": 0.02707,
    "R_er": 0.00369,
}

GENERIC_PARAMETERS: Dict[str, float] = {
    "C_env": 3.621417e6,
    "C_air": 3.953354e5,
    "R_rc": 0.007260759,
    "R_oe": 0.03000755,
    "R_er": 0.004009136,
}

# Broad positive bounds. These contain both the generic and true parameters.
PARAMETER_BOUNDS: Dict[str, Tuple[float, float]] = {
    "C_env": (1.0e6, 8.0e6),
    "C_air": (1.0e5, 1.0e6),
    "R_rc": (1.0e-3, 2.0e-2),
    "R_oe": (5.0e-3, 8.0e-2),
    "R_er": (5.0e-4, 2.0e-2),
}


# =============================================================================
# Plotting
# =============================================================================

def custom_plot(
    time,
    t_air,
    t_out,
    q_sg,
    sat_list,
    zat_list,
    lb_list,
    ub_list,
    phase_list,
    switch_time_hour,
    save_dir,
    file_stem="online_mpc_control_profile",
    show_plot=True,
):
    """Create the MPC control profile and mark the parameter-switch time."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    time = np.asarray(time, dtype=float)
    if time.size == 0:
        raise ValueError("Cannot plot an empty trajectory.")

    time_plot = time - time[0]
    relative_switch_time = None
    if switch_time_hour is not None:
        relative_switch_time = float(switch_time_hour) - float(time[0])

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
    ax_temp.grid(linestyle="--", linewidth=0.6, alpha=0.25)

    ax_ctrl.plot(time_plot, sat_list, linewidth=1.8, label="SAT")
    ax_ctrl.plot(time_plot, zat_list, linewidth=1.8, label="ZAT")
    ax_ctrl.set_title(
        "Control Inputs (SAT and ZAT)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax_ctrl.set_ylabel("Setpoint (°C)", fontsize=12)
    ax_ctrl.set_ylim(12.5, 26.5)
    ax_ctrl.legend(fontsize=10, loc="upper right")
    ax_ctrl.grid(linestyle="--", linewidth=0.6, alpha=0.25)

    ax_out.plot(time_plot, t_out, linewidth=1.6, alpha=0.85)
    ax_out.set_title(
        "Outdoor Air Temperature",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    ax_out.set_ylabel("Temperature (°C)", fontsize=12)
    ax_out.set_xlabel("Time (h)", fontsize=12)
    ax_out.set_xticks(ticks)
    ax_out.grid(linestyle="--", linewidth=0.6, alpha=0.25)

    ax_solar.plot(time_plot, q_sg, linewidth=1.6, alpha=0.85)
    ax_solar.set_title(
        "Solar Heat Gain",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    ax_solar.set_ylabel("Heat Gain (W)", fontsize=12)
    ax_solar.set_xlabel("Time (h)", fontsize=12)
    ax_solar.set_xticks(ticks)
    ax_solar.grid(linestyle="--", linewidth=0.6, alpha=0.25)

    if (
        relative_switch_time is not None
        and time_plot[0] <= relative_switch_time <= time_plot[-1]
    ):
        for ax in [ax_temp, ax_ctrl, ax_out, ax_solar]:
            ax.axvline(
                relative_switch_time,
                linestyle=":",
                linewidth=1.4,
                alpha=0.8,
            )
        ax_temp.text(
            relative_switch_time,
            ax_temp.get_ylim()[1],
            "  Identified model begins",
            va="top",
            ha="left",
            fontsize=8,
        )

    for ax in [ax_temp, ax_ctrl, ax_out, ax_solar]:
        ax.tick_params(axis="both", labelsize=10)

    ax_temp.tick_params(labelbottom=True)
    ax_ctrl.tick_params(labelbottom=True)

    plt.tight_layout()

    png_path = save_dir / f"{file_stem}.png"
    pdf_path = save_dir / f"{file_stem}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f">>> Saved plot to {png_path}")
    print(f">>> Saved plot to {pdf_path}")


# =============================================================================
# Helpers
# =============================================================================

def make_environment(
    *,
    data_file: str,
    start: float,
    end: float,
    parameters: Mapping[str, float],
) -> BEnv:
    """Create one environment using the supplied 3R2C parameters."""
    return BEnv(
        data_file=data_file,
        dt=1800.0,
        start=float(start),
        end=float(end),
        C_env=float(parameters["C_env"]),
        C_air=float(parameters["C_air"]),
        R_rc=float(parameters["R_rc"]),
        R_oe=float(parameters["R_oe"]),
        R_er=float(parameters["R_er"]),
        lb_set=22.0,
        ub_set=24.0,
    )


def make_controller(
    *,
    env,
    horizon: int,
    maxiter: int,
    ftol: float,
    sat_smooth_weight: float,
    zat_smooth_weight: float,
    default_sat: float,
    default_zat: float,
) -> MPCController:
    """Construct an MPC controller for the current prediction environment."""
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
    return controller


def relative_parameter_error_percent(
    estimate: Mapping[str, float],
    reference: Mapping[str, float],
) -> Dict[str, float]:
    return {
        name: 100.0
        * abs(float(estimate[name]) - float(reference[name]))
        / abs(float(reference[name]))
        for name in PARAMETER_NAMES
    }


def generate_noisy_weather_forecast(
    env,
    current_time: float,
    horizon: int,
    temperature_std: float = 0.8,
    solar_relative_std: float = 0.20,
    error_correlation: float = 0.8,
    rng=None,
) -> Dict[float, Dict[str, float]]:
    """
    Generate one fixed noisy weather forecast for a single MPC solve.

    The weather values stored in the environment dataset are treated as the
    forecast means. Correlated Gaussian forecast errors are then added:

        T_out_forecast = T_out_mean + temperature_error
        Qsg_forecast   = Qsg_mean * (1 + solar_relative_error)

    Adjacent horizon errors follow an AR(1) process so the forecast does not
    jump unrealistically between consecutive 30-minute points.

    The returned trajectory must be installed in the prediction environment
    once before controller.solve(...). It must remain fixed throughout that
    optimization call.

    Parameters
    ----------
    env:
        Current MPC prediction environment.
    current_time:
        Current real simulation time in absolute hours.
    horizon:
        Number of future MPC steps.
    temperature_std:
        Marginal standard deviation of outdoor-temperature forecast error
        in degrees Celsius.
    solar_relative_std:
        Marginal standard deviation of relative solar-gain forecast error.
        For example, 0.20 means 20 percent.
    error_correlation:
        AR(1) correlation between adjacent forecast errors.
    rng:
        NumPy random generator.

    Returns
    -------
    dict
        Mapping from absolute future time to forecast values and diagnostics.
    """
    horizon = int(horizon)
    temperature_std = float(temperature_std)
    solar_relative_std = float(solar_relative_std)
    rho = float(error_correlation)

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if temperature_std < 0.0:
        raise ValueError("temperature_std must be nonnegative.")
    if solar_relative_std < 0.0:
        raise ValueError("solar_relative_std must be nonnegative.")
    if not 0.0 <= rho < 1.0:
        raise ValueError("error_correlation must satisfy 0 <= rho < 1.")

    if rng is None:
        rng = np.random.default_rng()

    dt_hour = float(env.dt) / 3600.0
    innovation_scale = np.sqrt(max(0.0, 1.0 - rho ** 2))

    temperature_error = float(rng.normal(0.0, temperature_std))
    solar_relative_error = float(rng.normal(0.0, solar_relative_std))

    forecast: Dict[float, Dict[str, float]] = {}

    for forecast_step in range(1, horizon + 1):
        future_time = float(current_time) + forecast_step * dt_hour

        # Preserve the environment's original 30-minute dataset indexing.
        idx = min(
            max(int(float(future_time) * 2), 0),
            len(env.data) - 1,
        )
        row = env.data.iloc[idx]

        if forecast_step > 1:
            temperature_error = (
                rho * temperature_error
                + innovation_scale
                * float(rng.normal(0.0, temperature_std))
            )
            solar_relative_error = (
                rho * solar_relative_error
                + innovation_scale
                * float(rng.normal(0.0, solar_relative_std))
            )

        t_out_mean = float(row.Tout)
        qsg_mean = float(row.Qsg)

        t_out_forecast = t_out_mean + temperature_error
        qsg_forecast = max(
            qsg_mean * (1.0 + solar_relative_error),
            0.0,
        )

        forecast[round(future_time, 8)] = {
            "T_out": float(t_out_forecast),
            "Qsg": float(qsg_forecast),
            "T_out_mean": float(t_out_mean),
            "Qsg_mean": float(qsg_mean),
            "T_out_error": float(temperature_error),
            "Qsg_relative_error": float(solar_relative_error),
        }

    return forecast


# =============================================================================
# Main experiment
# =============================================================================

def run_online_mpc_test(
    start=17664.0,
    end=19872.5,
    horizon=6,
    maxiter=100,
    ftol=1e-3,
    sat_smooth_weight=0.0,
    zat_smooth_weight=0.0,
    default_sat=14.5,
    default_zat=23.0,
    identification_hours=24.0,
    identification_max_nfev=500,
    data_file="weather_data_2013_to_2017_summer_pandas.csv",
    show_plot=True,
    forecast_temperature_std=2.2,
    forecast_solar_relative_std=0.30,
    forecast_error_correlation=0.8,
    forecast_seed=42,
):
    """
    Run closed-loop online parameter-learning MPC.

    Real plant:
        Always uses TRUE_PARAMETERS.

    MPC prediction model:
        First identification_hours: GENERIC_PARAMETERS.
        Afterwards: parameters identified only from already observed data.

    Weather forecast used by MPC:
        The ground-truth future weather trajectory is treated as the forecast
        mean, then correlated random errors are added to T_out and Qsg.
        One fixed noisy forecast is generated per MPC solve.

    No future state, future measured transition, or true parameter is used by
    the controller or parameter estimator.
    """
    real_env = make_environment(
        data_file=data_file,
        start=float(start),
        end=float(end),
        parameters=TRUE_PARAMETERS,
    )
    real_env.reset()

    if float(forecast_temperature_std) < 0.0:
        raise ValueError("forecast_temperature_std must be nonnegative.")
    if float(forecast_solar_relative_std) < 0.0:
        raise ValueError("forecast_solar_relative_std must be nonnegative.")
    if not 0.0 <= float(forecast_error_correlation) < 1.0:
        raise ValueError(
            "forecast_error_correlation must satisfy 0 <= rho < 1."
        )

    forecast_rng = np.random.default_rng(int(forecast_seed))

    dt_hour = float(real_env.dt) / 3600.0
    identification_steps_float = float(identification_hours) / dt_hour
    identification_steps = int(round(identification_steps_float))

    if identification_steps < 1:
        raise ValueError("identification_hours must cover at least one step.")
    if not np.isclose(
        identification_steps * dt_hour,
        float(identification_hours),
        atol=1e-10,
    ):
        raise ValueError(
            "identification_hours must be an integer multiple of the outer "
            f"time step ({dt_hour} h)."
        )

    results_dir = Path("results") / "online_mpc"
    plots_dir = Path("plots") / "online_mpc"
    identification_dir = results_dir / "identification"

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    identification_dir.mkdir(parents=True, exist_ok=True)

    def prediction_env_factory(parameters: Mapping[str, float]):
        return make_environment(
            data_file=data_file,
            start=float(start),
            end=float(end),
            parameters=parameters,
        )

    adaptive_model = OnlineParameterLearningMPC(
        prediction_env_factory=prediction_env_factory,
        generic_parameters=GENERIC_PARAMETERS,
        parameter_bounds=PARAMETER_BOUNDS,
        identification_steps=identification_steps,
        envelope_weight=1.0,
        zone_weight=1.0,
        max_nfev=int(identification_max_nfev),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
        loss="linear",
        output_directory=identification_dir,
        verbose=True,
    )

    controller = make_controller(
        env=adaptive_model.prediction_env,
        horizon=horizon,
        maxiter=maxiter,
        ftol=ftol,
        sat_smooth_weight=sat_smooth_weight,
        zat_smooth_weight=zat_smooth_weight,
        default_sat=default_sat,
        default_zat=default_zat,
    )

    # Time-series records
    time_list: List[float] = []
    t_air_list: List[float] = []
    t_env_list: List[float] = []
    t_out_list: List[float] = []
    q_sg_list: List[float] = []
    q_int_list: List[float] = []

    # Forecast used for the first future step of each MPC solve.
    forecast_t_out_mean_list: List[float] = []
    forecast_t_out_sample_list: List[float] = []
    forecast_t_out_error_list: List[float] = []
    forecast_qsg_mean_list: List[float] = []
    forecast_qsg_sample_list: List[float] = []
    forecast_qsg_relative_error_list: List[float] = []

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

    mpc_phase_list: List[str] = []
    parameter_switched_after_step_list: List[bool] = []
    identification_rows_available_list: List[int] = []

    parameter_history: Dict[str, List[float]] = {
        name: [] for name in PARAMETER_NAMES
    }

    cumulative_energy = 0.0
    cumulative_exceedance = 0.0
    cumulative_violation_hours = 0.0

    total_steps_expected = int(np.ceil((end - start) / dt_hour))
    switch_time_hour = None
    step_idx = 0
    done = False

    print("=" * 78)
    print("ONLINE PARAMETER-LEARNING MPC TEST")
    print("=" * 78)
    print(f"Simulation interval:       {start:.3f} to {end:.3f} h")
    print(f"Outer step:               {dt_hour:.3f} h")
    print(f"Identification duration:  {identification_hours:.3f} h")
    print(f"Identification rows:      {identification_steps}")
    print("Real environment:         hidden true parameters")
    print("MPC during Day 1:         generic parameters")
    print("MPC after identification: inferred parameters")
    print(
        "Forecast temperature std: "
        f"{float(forecast_temperature_std):.3f} °C"
    )
    print(
        "Forecast solar rel. std:  "
        f"{100.0 * float(forecast_solar_relative_std):.2f}%"
    )
    print(
        "Forecast error corr.:     "
        f"{float(forecast_error_correlation):.3f}"
    )
    print(f"Forecast random seed:      {int(forecast_seed)}")
    print("=" * 78)

    while not done:
        step_idx += 1

        # The planning state comes from the real plant, but all predictions use
        # adaptive_model.prediction_env. This separates state observation from
        # model knowledge.
        start_snapshot = real_env.get_mpc_state()
        current_mpc_state = mpc_state_from_env(real_env)

        planning_phase = str(adaptive_model.phase)
        planning_parameters = adaptive_model.current_parameters.to_dict()

        adaptive_model.synchronize_prediction_env_time(real_env)

        # Generate exactly one fixed noisy forecast for this MPC solve.
        # SLSQP may evaluate the objective many times, but every evaluation
        # must see the same forecast trajectory.
        forecast = generate_noisy_weather_forecast(
            env=adaptive_model.prediction_env,
            current_time=float(real_env.t),
            horizon=int(horizon),
            temperature_std=float(forecast_temperature_std),
            solar_relative_std=float(forecast_solar_relative_std),
            error_correlation=float(forecast_error_correlation),
            rng=forecast_rng,
        )
        adaptive_model.prediction_env.set_forecast_override(forecast)

        try:
            action, diagnostics = controller.solve(
                initial_state=current_mpc_state
            )
        finally:
            # Prevent a forecast from leaking into a later solve or reset.
            adaptive_model.prediction_env.clear_forecast_override()

        first_forecast_time = round(float(real_env.t) + dt_hour, 8)
        first_forecast = forecast[first_forecast_time]

        _, reward, done, info = real_env.step(action)
        end_snapshot = real_env.get_mpc_state()

        controller.set_previous_action(action)

        switched_now = adaptive_model.record_transition(
            start_snapshot=start_snapshot,
            action=action,
            end_snapshot=end_snapshot,
            info=info,
            step_index=step_idx,
        )

        if switched_now:
            switch_time_hour = float(real_env.t)

            # Rebuild the controller so every future prediction uses the newly
            # created identified-parameter environment. Preserve the last real
            # action for optional smoothness penalties.
            previous_action = np.asarray(action, dtype=np.float64).copy()

            controller = make_controller(
                env=adaptive_model.prediction_env,
                horizon=horizon,
                maxiter=maxiter,
                ftol=ftol,
                sat_smooth_weight=sat_smooth_weight,
                zat_smooth_weight=zat_smooth_weight,
                default_sat=default_sat,
                default_zat=default_zat,
            )
            controller.set_previous_action(previous_action)

            print(
                f"\n>>> Parameter model switched after step {step_idx} "
                f"at t={real_env.t:.3f} h."
            )

        cumulative_energy += float(info["TotalEnergy_kWh"])

        exceed_step_degC_hr = float(info["TempExceed_degC"]) * dt_hour
        cumulative_exceedance += exceed_step_degC_hr

        if float(info["TempExceed_degC"]) > 0.0:
            cumulative_violation_hours += dt_hour

        time_list.append(float(real_env.t))
        t_air_list.append(float(info["T_zone_raw"]))
        t_env_list.append(float(info["T_env_raw"]))
        t_out_list.append(float(info["T_out_raw"]))
        q_sg_list.append(float(info["Qsg_raw"]))
        q_int_list.append(float(info["Qint_raw"]))

        forecast_t_out_mean_list.append(
            float(first_forecast["T_out_mean"])
        )
        forecast_t_out_sample_list.append(
            float(first_forecast["T_out"])
        )
        forecast_t_out_error_list.append(
            float(first_forecast["T_out_error"])
        )
        forecast_qsg_mean_list.append(
            float(first_forecast["Qsg_mean"])
        )
        forecast_qsg_sample_list.append(
            float(first_forecast["Qsg"])
        )
        forecast_qsg_relative_error_list.append(
            float(first_forecast["Qsg_relative_error"])
        )

        sat_list.append(float(info["SAT_sp"]))
        zat_list.append(float(info["ZAT_sp_used"]))
        lb_list.append(float(info["lb"]))
        ub_list.append(float(info["ub"]))

        reward_list.append(float(reward))
        energy_step_list.append(float(info["TotalEnergy_kWh"]))
        energy_cumulative_list.append(float(cumulative_energy))
        exceed_step_list.append(float(exceed_step_degC_hr))
        exceed_cumulative_list.append(float(cumulative_exceedance))
        violation_cumulative_list.append(float(cumulative_violation_hours))

        solver_success_list.append(bool(diagnostics.success))
        solver_fallback_list.append(bool(diagnostics.used_fallback))
        solver_time_list.append(float(diagnostics.solve_time_sec))
        solver_iterations_list.append(int(diagnostics.iterations))
        solver_nfev_list.append(int(diagnostics.function_evaluations))
        solver_objective_list.append(float(diagnostics.objective))
        solver_message_list.append(str(diagnostics.message))

        mpc_phase_list.append(planning_phase)
        parameter_switched_after_step_list.append(bool(switched_now))
        identification_rows_available_list.append(
            int(adaptive_model.collected_steps)
        )

        for name in PARAMETER_NAMES:
            parameter_history[name].append(float(planning_parameters[name]))

        if step_idx == 1 or step_idx % 10 == 0 or switched_now or done:
            print(
                f"[Online MPC {step_idx:04d}/{total_steps_expected:04d}] "
                f"phase={planning_phase:<10} | "
                f"t={real_env.t:.1f} h | "
                f"SAT={action[0]:.3f} | "
                f"ZAT={action[1]:.3f} | "
                f"Tzone={info['T_zone_raw']:.3f} °C | "
                f"E={cumulative_energy:.4f} kWh | "
                f"success={diagnostics.success} | "
                f"solve={diagnostics.solve_time_sec:.3f} s"
            )

    real_env.close()
    if hasattr(adaptive_model.prediction_env, "close"):
        adaptive_model.prediction_env.close()

    total_return = float(np.sum(reward_list))
    success_rate = float(np.mean(solver_success_list))
    fallback_rate = float(np.mean(solver_fallback_list))
    avg_solve_time = float(np.mean(solver_time_list))
    max_solve_time = float(np.max(solver_time_list))
    avg_iterations = float(np.mean(solver_iterations_list))
    avg_nfev = float(np.mean(solver_nfev_list))

    identified_parameters = (
        adaptive_model.identified_parameters.to_dict()
        if adaptive_model.identified_parameters is not None
        else {name: float("nan") for name in PARAMETER_NAMES}
    )
    identified_errors = relative_parameter_error_percent(
        identified_parameters,
        TRUE_PARAMETERS,
    )

    generic_steps = int(sum(phase == "generic" for phase in mpc_phase_list))
    identified_steps = int(sum(phase == "identified" for phase in mpc_phase_list))

    summary: Dict[str, Any] = {
        "method": "Online Identified MPC",
        "start_hour": float(start),
        "end_hour": float(end),
        "duration_hours": float(end - start),
        "horizon_steps": int(horizon),
        "horizon_hours": float(horizon * dt_hour),
        "total_steps": int(len(reward_list)),
        "identification_hours": float(identification_hours),
        "identification_steps": int(identification_steps),
        "forecast_temperature_std_degC": float(
            forecast_temperature_std
        ),
        "forecast_solar_relative_std": float(
            forecast_solar_relative_std
        ),
        "forecast_error_correlation": float(
            forecast_error_correlation
        ),
        "forecast_seed": int(forecast_seed),
        "generic_control_steps": generic_steps,
        "identified_control_steps": identified_steps,
        "parameter_switch_time_hour": switch_time_hour,
        "identification_elapsed_sec": (
            adaptive_model.identification_elapsed_seconds
        ),
        "identification_success": (
            None
            if adaptive_model.identification_result is None
            else bool(adaptive_model.identification_result.success)
        ),
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

    for name in PARAMETER_NAMES:
        summary[f"generic_{name}"] = float(GENERIC_PARAMETERS[name])
        summary[f"identified_{name}"] = float(identified_parameters[name])
        summary[f"true_{name}"] = float(TRUE_PARAMETERS[name])
        summary[f"identified_{name}_relative_error_percent"] = float(
            identified_errors[name]
        )

    profile_data: Dict[str, Any] = {
        "time_hour": time_list,
        "mpc_parameter_phase": mpc_phase_list,
        "parameter_switched_after_step": parameter_switched_after_step_list,
        "identification_rows_available": identification_rows_available_list,
        "Tenv_mpc": t_env_list,
        "Tair_mpc": t_air_list,
        "Tout": t_out_list,
        "Qsg": q_sg_list,
        "Qint": q_int_list,
        "forecast_Tout_mean_next_step": forecast_t_out_mean_list,
        "forecast_Tout_sample_next_step": forecast_t_out_sample_list,
        "forecast_Tout_error_next_step": forecast_t_out_error_list,
        "forecast_Qsg_mean_next_step": forecast_qsg_mean_list,
        "forecast_Qsg_sample_next_step": forecast_qsg_sample_list,
        "forecast_Qsg_relative_error_next_step": (
            forecast_qsg_relative_error_list
        ),
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
    for name in PARAMETER_NAMES:
        profile_data[f"mpc_model_{name}"] = parameter_history[name]

    profile_df = pd.DataFrame(profile_data)
    summary_df = pd.DataFrame([summary])

    profile_path = results_dir / "online_mpc_results_profile.csv"
    summary_path = results_dir / "online_mpc_results_summary.csv"
    parameter_path = results_dir / "online_identified_parameters.json"

    profile_df.to_csv(profile_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with parameter_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "generic_parameters": GENERIC_PARAMETERS,
                "identified_parameters": identified_parameters,
                "true_parameters_for_post_experiment_evaluation_only": (
                    TRUE_PARAMETERS
                ),
                "identified_relative_error_percent": identified_errors,
                "parameter_switch_time_hour": switch_time_hour,
                "identification_steps": identification_steps,
                "identification_hours": identification_hours,
                "weather_forecast_model": {
                    "temperature_error_distribution": "Gaussian",
                    "temperature_std_degC": float(
                        forecast_temperature_std
                    ),
                    "solar_relative_error_distribution": "Gaussian",
                    "solar_relative_std": float(
                        forecast_solar_relative_std
                    ),
                    "adjacent_error_correlation": float(
                        forecast_error_correlation
                    ),
                    "random_seed": int(forecast_seed),
                    "forecast_mean_source": (
                        "ground-truth future weather trajectory"
                    ),
                },
            },
            file,
            indent=2,
        )

    # Plot the same fixed paper window when available. For short smoke tests,
    # plot the full available trajectory instead of raising an exception.
    plot_start_idx = 2100
    plot_window = 400

    if len(time_list) > plot_start_idx:
        plot_end_idx = min(plot_start_idx + plot_window, len(time_list))
    else:
        plot_start_idx = 0
        plot_end_idx = len(time_list)

    custom_plot(
        time=np.asarray(time_list)[plot_start_idx:plot_end_idx],
        t_air=np.asarray(t_air_list)[plot_start_idx:plot_end_idx],
        t_out=np.asarray(t_out_list)[plot_start_idx:plot_end_idx],
        q_sg=np.asarray(q_sg_list)[plot_start_idx:plot_end_idx],
        sat_list=np.asarray(sat_list)[plot_start_idx:plot_end_idx],
        zat_list=np.asarray(zat_list)[plot_start_idx:plot_end_idx],
        lb_list=np.asarray(lb_list)[plot_start_idx:plot_end_idx],
        ub_list=np.asarray(ub_list)[plot_start_idx:plot_end_idx],
        phase_list=np.asarray(mpc_phase_list)[plot_start_idx:plot_end_idx],
        switch_time_hour=switch_time_hour,
        save_dir=plots_dir,
        file_stem="online_mpc_control_profile",
        show_plot=show_plot,
    )

    print("\n" + "=" * 78)
    print("ONLINE MPC TEST SUMMARY")
    print("=" * 78)
    print(f"Simulation interval:        {start:.1f} to {end:.1f} h")
    print(
        f"Prediction horizon:         {horizon} steps "
        f"({horizon * dt_hour:.1f} h)"
    )
    print(f"Generic-control steps:      {generic_steps}")
    print(f"Identified-control steps:   {identified_steps}")
    print(f"Parameter switch time:      {switch_time_hour}")
    print(
        "Identification time:       "
        f"{adaptive_model.identification_elapsed_seconds}"
    )
    print(f"Total Return:               {total_return:.8f}")
    print(f"Energy Use:                 {cumulative_energy:.6f} kWh")
    print(
        "Hours out of Bounds:       "
        f"{cumulative_violation_hours:.4f}"
    )
    print(
        "Temperature Exceedance:    "
        f"{cumulative_exceedance:.6f} degC-hr"
    )
    print(f"Solver Success Rate:        {100.0 * success_rate:.2f}%")
    print(f"Fallback Rate:              {100.0 * fallback_rate:.2f}%")
    print(f"Average Solver Time:        {avg_solve_time:.6f} s")
    print(f"Maximum Solver Time:        {max_solve_time:.6f} s")

    print("\nIdentified parameters")
    print("-" * 78)
    for name in PARAMETER_NAMES:
        print(
            f"{name:<8} = {identified_parameters[name]:.12g} | "
            f"true = {TRUE_PARAMETERS[name]:.12g} | "
            f"error = {identified_errors[name]:.6g}%"
        )

    print(f"\n>>> Saved profile to {profile_path}")
    print(f">>> Saved summary to {summary_path}")
    print(f">>> Saved parameters to {parameter_path}")
    print("=" * 78)

    return summary, profile_df


# Backward-compatible alias for callers that previously imported run_mpc_test.
run_mpc_test = run_online_mpc_test


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run closed-loop online parameter-learning MPC: use generic "
            "parameters first, identify from past transitions, then switch "
            "the MPC prediction model."
        )
    )

    parser.add_argument("--start", type=float, default=17664.0)
    parser.add_argument("--end", type=float, default=19872.5)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--ftol", type=float, default=3.37e-6)

    parser.add_argument("--sat_smooth_weight", type=float, default=0.0)
    parser.add_argument("--zat_smooth_weight", type=float, default=0.0)

    parser.add_argument("--default_sat", type=float, default=14.5)
    parser.add_argument("--default_zat", type=float, default=23.0)

    parser.add_argument(
        "--identification_hours",
        type=float,
        default=24.0,
        help=(
            "Hours of already observed transitions used before the one-time "
            "parameter update. With dt=1800 s, 24 h means 48 rows."
        ),
    )
    parser.add_argument(
        "--identification_max_nfev",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="weather_data_2013_to_2017_summer_pandas.csv",
    )

    parser.add_argument(
        "--forecast_temperature_std",
        type=float,
        default=0.8,
        help=(
            "Standard deviation of outdoor-temperature forecast error "
            "in degrees Celsius."
        ),
    )
    parser.add_argument(
        "--forecast_solar_relative_std",
        type=float,
        default=0.20,
        help=(
            "Standard deviation of relative solar-gain forecast error. "
            "For example, 0.20 means 20 percent."
        ),
    )
    parser.add_argument(
        "--forecast_error_correlation",
        type=float,
        default=0.8,
        help=(
            "AR(1) correlation between adjacent forecast errors."
        ),
    )
    parser.add_argument(
        "--forecast_seed",
        type=int,
        default=35,
        help="Random seed used to generate noisy weather forecasts.",
    )

    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Save plots without opening a window.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_online_mpc_test(
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        maxiter=args.maxiter,
        ftol=args.ftol,
        sat_smooth_weight=args.sat_smooth_weight,
        zat_smooth_weight=args.zat_smooth_weight,
        default_sat=args.default_sat,
        default_zat=args.default_zat,
        identification_hours=args.identification_hours,
        identification_max_nfev=args.identification_max_nfev,
        data_file=args.data_file,
        show_plot=not args.no_plot,
        forecast_temperature_std=args.forecast_temperature_std,
        forecast_solar_relative_std=args.forecast_solar_relative_std,
        forecast_error_correlation=args.forecast_error_correlation,
        forecast_seed=args.forecast_seed,
    )


if __name__ == "__main__":
    main()
