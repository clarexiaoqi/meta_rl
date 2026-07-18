"""
collect_identification_data.py

Collect transition data for 3R2C parameter identification.

This version records all information required to reproduce one complete
outer environment transition exactly, including:

- Initial envelope and zone temperatures
- SAT and ZAT actions
- Initial PI-controller states
- Disturbances actually used by the environment
- Final envelope and zone temperatures
- Final PI-controller states
- Additional HVAC and energy information for diagnostics

The identification model should later use the same six internal PI loops
and the same exact discrete-time state-space model as the environment.

Important
---------
Q_HVAC_total_est_W is no longer used as the main identification input.
The HVAC heat flow must be reconstructed inside the identification model
from the action and initial PI-controller state, exactly as in the
environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

ENVIRONMENT_MODULE = "Env_develop_mpc_draft"
ENVIRONMENT_CLASS = "ContinuousBuildingControlEnvironment"

WEATHER_DATA_FILE = (
    "weather_data_2013_to_2017_summer_pandas.csv"
)

# The weather CSV itself is read by the environment from ./data/.
START_TIME_HOUR = 17664.0

# Seven days × 48 transitions/day = 336 transitions.
COLLECTION_DAYS = 7.0
DT_SECONDS = 1800.0

END_TIME_HOUR = (
    START_TIME_HOUR + 24.0 * COLLECTION_DAYS
)

OUTPUT_FILE = (
    Path("results")
    / "identification"
    / "identification_data.csv"
)

METADATA_FILE = (
    Path("results")
    / "identification"
    / "identification_data_metadata.json"
)

RANDOM_SEED = 42


# ============================================================
# True environment parameters
# ============================================================

TRUE_PARAMETERS = {
    "C_env": 3.1996e6,
    "C_air": 3.5187e5,
    "R_rc": 0.00706,
    "R_oe": 0.02707,
    "R_er": 0.00369,
}


# ============================================================
# Environment settings
# ============================================================

ENVIRONMENT_KWARGS = {
    "data_file": WEATHER_DATA_FILE,
    "dt": DT_SECONDS,
    "start": START_TIME_HOUR,
    "end": END_TIME_HOUR,

    "C_env": TRUE_PARAMETERS["C_env"],
    "C_air": TRUE_PARAMETERS["C_air"],
    "R_rc": TRUE_PARAMETERS["R_rc"],
    "R_oe": TRUE_PARAMETERS["R_oe"],
    "R_er": TRUE_PARAMETERS["R_er"],

    "lb_set": 22.0,
    "ub_set": 24.0,
    "night_lb_set": 15.0,
    "night_ub_set": 28.0,

    "mode_deadband": 0.0,
    "alpha": 0.3,
    "E_ref": 60.0,
    "T_ref": 8.0,

    "SAT_low": 12.8,
    "SAT_high": 17.7,
    "ZAT_low": 18.0,
    "ZAT_high": 26.0,

    "Qh_reheat_max": 300.0,
}


# ============================================================
# Excitation action configuration
# ============================================================

# Holding each action for several outer steps avoids changing the
# setpoint every 30 minutes, while still exciting different operating
# conditions.
ACTION_HOLD_STEPS_MIN = 2
ACTION_HOLD_STEPS_MAX = 6

SAT_MIN = 12.8
SAT_MAX = 17.7

ZAT_MIN = 18.0
ZAT_MAX = 26.0


# ============================================================
# Utility functions
# ============================================================

def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and environment action seeds."""

    random.seed(seed)
    np.random.seed(seed)


def load_environment_class():
    """Dynamically load the environment class."""

    module = importlib.import_module(
        ENVIRONMENT_MODULE
    )

    if not hasattr(module, ENVIRONMENT_CLASS):
        raise AttributeError(
            f"Module '{ENVIRONMENT_MODULE}' does not contain "
            f"'{ENVIRONMENT_CLASS}'."
        )

    return getattr(module, ENVIRONMENT_CLASS)


def create_environment():
    """Create the true-parameter data-generating environment."""

    environment_class = load_environment_class()

    env = environment_class(
        **ENVIRONMENT_KWARGS
    )

    env.seed(RANDOM_SEED)

    return env


def optional_float(value: Any) -> float:
    """
    Convert an optional scalar to a CSV-compatible float.

    None is stored as NaN. This is especially important for the initial
    prev_ZAT_sp value after env.reset().
    """

    if value is None:
        return float("nan")

    return float(value)


def generate_excitation_action(
    rng: np.random.Generator,
    step_index: int,
    current_action: Optional[np.ndarray],
    remaining_hold_steps: int,
) -> tuple[np.ndarray, int]:
    """
    Generate a deterministic, persistently exciting SAT/ZAT action.

    The action is held for 2-6 outer steps and then resampled.

    A mixture of boundary, middle, and random actions is used so that
    the data cover cooling, neutral, and possible reheat conditions.
    """

    if (
        current_action is not None
        and remaining_hold_steps > 0
    ):
        return (
            current_action.copy(),
            remaining_hold_steps - 1,
        )

    structured_actions = np.array(
        [
            [12.8, 18.0],
            [12.8, 22.0],
            [12.8, 24.0],
            [12.8, 26.0],

            [14.5, 20.0],
            [14.5, 22.0],
            [14.5, 24.0],
            [14.5, 26.0],

            [16.0, 20.0],
            [16.0, 22.0],
            [16.0, 24.0],
            [16.0, 26.0],

            [17.7, 18.0],
            [17.7, 22.0],
            [17.7, 24.0],
            [17.7, 26.0],
        ],
        dtype=np.float64,
    )

    # Most action blocks use a structured excitation point. Some use
    # uniformly sampled points to avoid repeatedly visiting only a grid.
    if rng.random() < 0.75:
        action_index = int(
            rng.integers(
                0,
                len(structured_actions),
            )
        )
        action = structured_actions[
            action_index
        ].copy()
    else:
        action = np.array(
            [
                rng.uniform(SAT_MIN, SAT_MAX),
                rng.uniform(ZAT_MIN, ZAT_MAX),
            ],
            dtype=np.float64,
        )

    hold_steps = int(
        rng.integers(
            ACTION_HOLD_STEPS_MIN,
            ACTION_HOLD_STEPS_MAX + 1,
        )
    )

    return action, hold_steps - 1


def validate_snapshot(
    snapshot: Dict[str, Any],
) -> None:
    """Validate the state snapshot returned by get_mpc_state()."""

    required_keys = {
        "raw_state",
        "time_hour",
        "integral_error",
        "prev_zat_sp",
        "damper_signal_prev",
        "m_fan",
    }

    missing_keys = (
        required_keys - set(snapshot.keys())
    )

    if missing_keys:
        raise KeyError(
            "get_mpc_state() is missing keys: "
            f"{sorted(missing_keys)}"
        )

    raw_state = np.asarray(
        snapshot["raw_state"],
        dtype=np.float64,
    ).reshape(-1)

    if raw_state.size != 8:
        raise ValueError(
            "Expected raw_state with 8 entries, "
            f"but received shape {raw_state.shape}."
        )


def validate_info(
    info: Dict[str, Any],
) -> None:
    """Validate the environment information dictionary."""

    required_keys = {
        "SAT_sp",
        "ZAT_sp_used",

        "T_env_raw",
        "T_zone_raw",
        "T_cor_raw",
        "T_out_raw",
        "Qsg_raw",
        "Qint_raw",

        "m_fan",
        "DamperSignal",

        "Q_reheat_W",
        "TotalEnergy_kWh",
    }

    missing_keys = (
        required_keys - set(info.keys())
    )

    if missing_keys:
        raise KeyError(
            "Environment info is missing keys: "
            f"{sorted(missing_keys)}"
        )


def make_transition_row(
    step_index: int,
    snapshot: Dict[str, Any],
    requested_action: np.ndarray,
    next_observation: np.ndarray,
    reward: float,
    done: bool,
    info: Dict[str, Any],
    env,
) -> Dict[str, Any]:
    """
    Build one identification-data row.

    The snapshot is taken immediately before env.step(). The info and
    environment controller states are read immediately after env.step().
    """

    validate_snapshot(snapshot)
    validate_info(info)

    raw_state = np.asarray(
        snapshot["raw_state"],
        dtype=np.float64,
    ).reshape(-1)

    requested_action = np.asarray(
        requested_action,
        dtype=np.float64,
    ).reshape(-1)

    if requested_action.size != 2:
        raise ValueError(
            "Expected requested action [SAT, ZAT], "
            f"got shape {requested_action.shape}."
        )

    next_observation = np.asarray(
        next_observation,
        dtype=np.float64,
    ).reshape(-1)

    if next_observation.size != 8:
        raise ValueError(
            "Expected next observation with 8 entries, "
            f"got shape {next_observation.shape}."
        )

    time_hour_start = float(
        snapshot["time_hour"]
    )

    time_hour_end = float(env.t)

    expected_time_end = (
        time_hour_start + DT_SECONDS / 3600.0
    )

    if not np.isclose(
        time_hour_end,
        expected_time_end,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Unexpected environment time increment: "
            f"start={time_hour_start}, "
            f"end={time_hour_end}, "
            f"expected={expected_time_end}."
        )

    # Start-state fields correspond to:
    # [T_env, T_zone, T_cor, T_out, Qsg, Qint, hour_sin, hour_cos]
    row = {
        # ----------------------------------------------------
        # Transition identifiers
        # ----------------------------------------------------
        "row_index": int(step_index),
        "time_hour_start": time_hour_start,
        "time_hour_end": time_hour_end,
        "dt_seconds": float(DT_SECONDS),

        # ----------------------------------------------------
        # Initial thermal and disturbance state
        # ----------------------------------------------------
        "T_env_start": float(raw_state[0]),
        "T_zone_start": float(raw_state[1]),
        "T_cor_state_start": float(raw_state[2]),
        "T_out_state_start": float(raw_state[3]),
        "Qsg_state_start": float(raw_state[4]),
        "Qint_state_start": float(raw_state[5]),
        "hour_sin_start": float(raw_state[6]),
        "hour_cos_start": float(raw_state[7]),

        # ----------------------------------------------------
        # Requested and clipped action
        # ----------------------------------------------------
        "SAT_sp_requested": float(
            requested_action[0]
        ),
        "ZAT_sp_requested": float(
            requested_action[1]
        ),

        # These are the actual clipped values used by the env.
        "SAT_sp": float(info["SAT_sp"]),
        "ZAT_sp": float(info["ZAT_sp_used"]),

        # ----------------------------------------------------
        # Initial PI-controller state
        # ----------------------------------------------------
        "integral_error_start": float(
            snapshot["integral_error"]
        ),
        "prev_zat_sp_start": optional_float(
            snapshot["prev_zat_sp"]
        ),
        "damper_signal_prev_start": float(
            snapshot["damper_signal_prev"]
        ),
        "m_fan_start": float(
            snapshot["m_fan"]
        ),

        # ----------------------------------------------------
        # Disturbances actually used for this transition
        # ----------------------------------------------------
        # The environment advances the clock first and then reads
        # these values. They are held constant during all six PI loops.
        "T_cor_used": float(info["T_cor_raw"]),
        "T_out_used": float(info["T_out_raw"]),
        "Qsg_used": float(info["Qsg_raw"]),
        "Qint_used": float(info["Qint_raw"]),
        "Hour_used": float(info["Hour"]),

        # ----------------------------------------------------
        # Final measured thermal state
        # ----------------------------------------------------
        "T_env_end": float(info["T_env_raw"]),
        "T_zone_end": float(info["T_zone_raw"]),

        # ----------------------------------------------------
        # Final PI-controller state
        # ----------------------------------------------------
        "integral_error_end": float(
            env.integral_error
        ),
        "prev_zat_sp_end": optional_float(
            env.prev_ZAT_sp
        ),
        "damper_signal_end": float(
            env.damper_signal_prev
        ),
        "m_fan_end": float(
            env.m_fan
        ),

        # ----------------------------------------------------
        # Final-step HVAC diagnostics
        # ----------------------------------------------------
        # These values are not sufficient by themselves to reproduce
        # the full transition because they describe only the final
        # internal PI iteration. They are retained for diagnostics.
        "Mode": str(info["Mode"]),
        "ReheatSignal_end": float(
            info["ReheatSignal"]
        ),
        "Q_reheat_end_W": float(
            info["Q_reheat_W"]
        ),

        # ----------------------------------------------------
        # Energy and comfort diagnostics
        # ----------------------------------------------------
        "reward": float(reward),
        "done": bool(done),

        "TotalEnergy_kWh": float(
            info["TotalEnergy_kWh"]
        ),
        "CoolingEnergy_kWh": float(
            info["CoolingEnergy_kWh"]
        ),
        "HeatingEnergy_kWh": float(
            info["HeatingEnergy_kWh"]
        ),
        "ReheatEnergy_kWh": float(
            info["ReheatEnergy_kWh"]
        ),
        "FanEnergy_kWh": float(
            info["FanEnergy_kWh"]
        ),

        "ComfortPeriod": str(
            info["ComfortPeriod"]
        ),
        "ComfortLB": float(info["ComfortLB"]),
        "ComfortUB": float(info["ComfortUB"]),
        "TempExceed_degC": float(
            info["TempExceed_degC"]
        ),

        # ----------------------------------------------------
        # Stored normalized next observation for consistency checks
        # ----------------------------------------------------
        "T_env_end_normalized": float(
            next_observation[0]
        ),
        "T_zone_end_normalized": float(
            next_observation[1]
        ),
    }

    return row


def check_transition_consistency(
    dataframe: pd.DataFrame,
) -> Dict[str, float]:
    """
    Check continuity and basic consistency of the collected trajectory.
    """

    if dataframe.empty:
        raise ValueError(
            "The collected dataframe is empty."
        )

    if len(dataframe) == 1:
        return {
            "max_T_env_continuity_error": 0.0,
            "max_T_zone_continuity_error": 0.0,
            "max_time_continuity_error": 0.0,
        }

    env_continuity_error = (
        dataframe["T_env_start"]
        .iloc[1:]
        .to_numpy(dtype=np.float64)
        - dataframe["T_env_end"]
        .iloc[:-1]
        .to_numpy(dtype=np.float64)
    )

    zone_continuity_error = (
        dataframe["T_zone_start"]
        .iloc[1:]
        .to_numpy(dtype=np.float64)
        - dataframe["T_zone_end"]
        .iloc[:-1]
        .to_numpy(dtype=np.float64)
    )

    time_continuity_error = (
        dataframe["time_hour_start"]
        .iloc[1:]
        .to_numpy(dtype=np.float64)
        - dataframe["time_hour_end"]
        .iloc[:-1]
        .to_numpy(dtype=np.float64)
    )

    return {
        "max_T_env_continuity_error": float(
            np.max(np.abs(env_continuity_error))
        ),
        "max_T_zone_continuity_error": float(
            np.max(np.abs(zone_continuity_error))
        ),
        "max_time_continuity_error": float(
            np.max(np.abs(time_continuity_error))
        ),
    }


def save_metadata(
    dataframe: pd.DataFrame,
    consistency_metrics: Dict[str, float],
) -> None:
    """Save collection settings and a short dataset summary."""

    metadata = {
        "environment_module":
            ENVIRONMENT_MODULE,
        "environment_class":
            ENVIRONMENT_CLASS,
        "weather_data_file":
            WEATHER_DATA_FILE,

        "random_seed":
            RANDOM_SEED,
        "start_time_hour":
            START_TIME_HOUR,
        "end_time_hour":
            END_TIME_HOUR,
        "dt_seconds":
            DT_SECONDS,
        "number_of_rows":
            int(len(dataframe)),

        "true_parameters":
            TRUE_PARAMETERS,

        "action_ranges": {
            "SAT_min": SAT_MIN,
            "SAT_max": SAT_MAX,
            "ZAT_min": ZAT_MIN,
            "ZAT_max": ZAT_MAX,
            "hold_steps_min":
                ACTION_HOLD_STEPS_MIN,
            "hold_steps_max":
                ACTION_HOLD_STEPS_MAX,
        },

        "dataset_ranges": {
            "T_env_start_min": float(
                dataframe["T_env_start"].min()
            ),
            "T_env_start_max": float(
                dataframe["T_env_start"].max()
            ),
            "T_zone_start_min": float(
                dataframe["T_zone_start"].min()
            ),
            "T_zone_start_max": float(
                dataframe["T_zone_start"].max()
            ),
            "SAT_sp_min": float(
                dataframe["SAT_sp"].min()
            ),
            "SAT_sp_max": float(
                dataframe["SAT_sp"].max()
            ),
            "ZAT_sp_min": float(
                dataframe["ZAT_sp"].min()
            ),
            "ZAT_sp_max": float(
                dataframe["ZAT_sp"].max()
            ),
            "Qsg_used_min": float(
                dataframe["Qsg_used"].min()
            ),
            "Qsg_used_max": float(
                dataframe["Qsg_used"].max()
            ),
            "Qint_used_min": float(
                dataframe["Qint_used"].min()
            ),
            "Qint_used_max": float(
                dataframe["Qint_used"].max()
            ),
        },

        "consistency_metrics":
            consistency_metrics,

        "identification_note": (
            "The identification model must reconstruct the six internal "
            "PI loops and the exact 300-second state-space discretization. "
            "Final m_fan and Q_reheat values are diagnostics only."
        ),
    }

    METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main data collection
# ============================================================

def collect_identification_data(
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Run the true environment and collect transition data."""

    set_random_seed(RANDOM_SEED)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    env = create_environment()

    if not hasattr(env, "get_mpc_state"):
        raise AttributeError(
            "The environment must implement get_mpc_state()."
        )

    observation = env.reset()

    expected_steps = int(
        round(
            (
                END_TIME_HOUR
                - START_TIME_HOUR
            )
            * 3600.0
            / DT_SECONDS
        )
    )

    print("=" * 78)
    print("COLLECTING PARAMETER-IDENTIFICATION DATA")
    print("=" * 78)
    print(
        f"Environment:      "
        f"{ENVIRONMENT_MODULE}.{ENVIRONMENT_CLASS}"
    )
    print(
        f"Weather file:     {WEATHER_DATA_FILE}"
    )
    print(
        f"Start time:       {START_TIME_HOUR:.3f} h"
    )
    print(
        f"End time:         {END_TIME_HOUR:.3f} h"
    )
    print(
        f"Outer time step:  {DT_SECONDS:.1f} s"
    )
    print(
        f"Expected rows:    {expected_steps}"
    )
    print(
        f"Output file:      {output_file}"
    )
    print("=" * 78)
    print()

    rows: list[Dict[str, Any]] = []

    current_action: Optional[
        np.ndarray
    ] = None

    remaining_hold_steps = 0

    done = False
    step_index = 0

    try:
        while not done:
            if step_index >= expected_steps:
                # Normally the final environment step should already set
                # done=True. This guard prevents accidental extra rows.
                break

            snapshot = env.get_mpc_state()

            (
                current_action,
                remaining_hold_steps,
            ) = generate_excitation_action(
                rng=rng,
                step_index=step_index,
                current_action=current_action,
                remaining_hold_steps=(
                    remaining_hold_steps
                ),
            )

            action = current_action.astype(
                np.float32
            )

            (
                next_observation,
                reward,
                done,
                info,
            ) = env.step(action)

            row = make_transition_row(
                step_index=step_index,
                snapshot=snapshot,
                requested_action=action,
                next_observation=(
                    next_observation
                ),
                reward=reward,
                done=done,
                info=info,
                env=env,
            )

            rows.append(row)

            observation = (
                np.asarray(
                    next_observation,
                    dtype=np.float32,
                ).copy()
            )

            if (
                step_index == 0
                or (step_index + 1) % 48 == 0
                or done
            ):
                print(
                    f"Step {step_index + 1:4d}/"
                    f"{expected_steps} | "
                    f"t={env.t:.2f} h | "
                    f"SAT={row['SAT_sp']:.2f} °C | "
                    f"ZAT={row['ZAT_sp']:.2f} °C | "
                    f"T_zone={row['T_zone_end']:.3f} °C"
                )

            step_index += 1

    finally:
        close_method = getattr(
            env,
            "close",
            None,
        )

        if callable(close_method):
            close_method()

    dataframe = pd.DataFrame(rows)

    if len(dataframe) != expected_steps:
        raise RuntimeError(
            f"Expected {expected_steps} transitions, "
            f"but collected {len(dataframe)}."
        )

    consistency_metrics = (
        check_transition_consistency(
            dataframe
        )
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_file,
        index=False,
        float_format="%.12g",
    )

    save_metadata(
        dataframe=dataframe,
        consistency_metrics=(
            consistency_metrics
        ),
    )

    print()
    print("=" * 78)
    print("COLLECTION COMPLETE")
    print("=" * 78)
    print(
        f"Rows collected:   {len(dataframe)}"
    )
    print(
        f"Columns saved:    {len(dataframe.columns)}"
    )
    print(
        f"Dataset saved:    {output_file}"
    )
    print(
        f"Metadata saved:   {METADATA_FILE}"
    )
    print()

    print("Trajectory consistency")
    print("-" * 78)
    print(
        "Maximum T_env continuity error:  "
        f"{consistency_metrics['max_T_env_continuity_error']:.12g} °C"
    )
    print(
        "Maximum T_zone continuity error: "
        f"{consistency_metrics['max_T_zone_continuity_error']:.12g} °C"
    )
    print(
        "Maximum time continuity error:   "
        f"{consistency_metrics['max_time_continuity_error']:.12g} h"
    )
    print()

    print("Action coverage")
    print("-" * 78)
    print(
        "SAT range: "
        f"[{dataframe['SAT_sp'].min():.3f}, "
        f"{dataframe['SAT_sp'].max():.3f}] °C"
    )
    print(
        "ZAT range: "
        f"[{dataframe['ZAT_sp'].min():.3f}, "
        f"{dataframe['ZAT_sp'].max():.3f}] °C"
    )
    print()

    print("Thermal-state coverage")
    print("-" * 78)
    print(
        "T_env range:  "
        f"[{dataframe['T_env_start'].min():.3f}, "
        f"{dataframe['T_env_start'].max():.3f}] °C"
    )
    print(
        "T_zone range: "
        f"[{dataframe['T_zone_start'].min():.3f}, "
        f"{dataframe['T_zone_start'].max():.3f}] °C"
    )
    print("=" * 78)

    return dataframe


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Parse optional command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Collect exact outer-transition data for "
            "3R2C parameter identification."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=(
            "Output CSV path. Default: "
            f"{OUTPUT_FILE}"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    arguments = parse_arguments()

    collect_identification_data(
        output_file=arguments.output,
    )


if __name__ == "__main__":
    main()