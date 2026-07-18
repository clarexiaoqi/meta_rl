from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MPCState:
    """
    Complete prediction state required by the MPC rollout.

    Parameters
    ----------
    raw_state:
        Denormalized environment state with shape (8,):
        [T_env, T_zone, T_cor, T_out, Qsg, Qint, hour_sin, hour_cos].
    time_hour:
        Current absolute simulation time in hours.
    integral_error:
        PI controller integral state.
    prev_zat_sp:
        ZAT setpoint used in the previous outer environment step.
    damper_signal_prev:
        Damper signal carried from the previous outer environment step.
    m_fan:
        Most recently calculated fan mass flow rate.
    """

    raw_state: np.ndarray
    time_hour: float
    integral_error: float
    prev_zat_sp: Optional[float]
    damper_signal_prev: float
    m_fan: float

    def copy(self) -> "MPCState":
        return MPCState(
            raw_state=np.asarray(self.raw_state, dtype=np.float64).copy(),
            time_hour=float(self.time_hour),
            integral_error=float(self.integral_error),
            prev_zat_sp=(
                None if self.prev_zat_sp is None else float(self.prev_zat_sp)
            ),
            damper_signal_prev=float(self.damper_signal_prev),
            m_fan=float(self.m_fan),
        )


@dataclass
class MPCStepRecord:
    """One predicted outer-step result."""

    action: np.ndarray
    reward: float
    cost: float
    energy_kwh: float
    temp_exceed_degC: float
    comfort_lb: float
    comfort_ub: float
    t_env: float
    t_zone: float
    t_out: float
    qsg: float
    qint: float
    hour: float
    damper_signal: float
    m_fan: float
    solver_metadata: Optional[Dict[str, Any]] = None


@dataclass
class MPCRolloutResult:
    """Complete multi-step prediction result."""

    total_cost: float
    total_reward: float
    total_energy_kwh: float
    total_temp_exceed_degC_hr: float
    violation_hours: float
    final_state: MPCState
    records: List[MPCStepRecord]


def mpc_state_from_env(env) -> MPCState:
    """
    Build an MPCState snapshot from the current real environment.

    The environment must implement get_mpc_state().
    """
    snapshot = env.get_mpc_state()

    return MPCState(
        raw_state=np.asarray(snapshot["raw_state"], dtype=np.float64).copy(),
        time_hour=float(snapshot["time_hour"]),
        integral_error=float(snapshot["integral_error"]),
        prev_zat_sp=(
            None
            if snapshot["prev_zat_sp"] is None
            else float(snapshot["prev_zat_sp"])
        ),
        damper_signal_prev=float(snapshot["damper_signal_prev"]),
        m_fan=float(snapshot["m_fan"]),
    )


def _validate_action(env, action: Sequence[float]) -> np.ndarray:
    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)

    if action_arr.size != 2:
        raise ValueError(
            f"Each MPC action must contain [SAT, ZAT], got shape {action_arr.shape}."
        )

    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)

    return np.clip(action_arr, low, high)


def _validate_action_sequence(
    env,
    action_sequence: Sequence[Sequence[float]],
) -> np.ndarray:
    actions = np.asarray(action_sequence, dtype=np.float64)

    if actions.ndim == 1:
        if actions.size % 2 != 0:
            raise ValueError(
                "A flattened action sequence must contain an even number of values."
            )
        actions = actions.reshape(-1, 2)

    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(
            "action_sequence must have shape (horizon, 2) for [SAT, ZAT]."
        )

    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)

    return np.clip(actions, low, high)


def propagate_one_step(
    env,
    state: MPCState,
    action: Sequence[float],
) -> Tuple[MPCState, MPCStepRecord]:
    """
    Predict one outer environment step without modifying the real environment.

    This function calls env.predict_step(...), then converts its dictionary
    output into structured MPCState and MPCStepRecord objects.
    """
    action_arr = _validate_action(env, action)

    result = env.predict_step(
        action=action_arr,
        raw_state=state.raw_state,
        time_hour=state.time_hour,
        integral_error=state.integral_error,
        prev_zat_sp=state.prev_zat_sp,
        damper_signal_prev=state.damper_signal_prev,
    )

    info = result["info"]

    next_state = MPCState(
        raw_state=np.asarray(result["next_raw_state"], dtype=np.float64).copy(),
        time_hour=float(result["next_time"]),
        integral_error=float(result["next_integral_error"]),
        prev_zat_sp=float(result["next_prev_zat_sp"]),
        damper_signal_prev=float(result["next_damper_signal"]),
        m_fan=float(result["next_m_fan"]),
    )

    cost = -float(result["reward"])

    record = MPCStepRecord(
        action=action_arr.copy(),
        reward=float(result["reward"]),
        cost=cost,
        energy_kwh=float(info["TotalEnergy_kWh"]),
        temp_exceed_degC=float(info["TempExceed_degC"]),
        comfort_lb=float(info["ComfortLB"]),
        comfort_ub=float(info["ComfortUB"]),
        t_env=float(info["T_env_raw"]),
        t_zone=float(info["T_zone_raw"]),
        t_out=float(info["T_out_raw"]),
        qsg=float(info["Qsg_raw"]),
        qint=float(info["Qint_raw"]),
        hour=float(info["Hour"]),
        damper_signal=float(info["DamperSignal"]),
        m_fan=float(info["m_fan"]),
    )

    return next_state, record


def rollout_prediction(
    env,
    initial_state: MPCState,
    action_sequence: Sequence[Sequence[float]],
    sat_smooth_weight: float = 0.0,
    zat_smooth_weight: float = 0.0,
    previous_action: Optional[Sequence[float]] = None,
) -> MPCRolloutResult:
    """
    Predict a complete MPC horizon.

    Parameters
    ----------
    env:
        MPC-compatible environment implementing predict_step().
    initial_state:
        Current thermal and PI-controller state.
    action_sequence:
        Array-like with shape (horizon, 2), where each row is [SAT, ZAT].
    sat_smooth_weight:
        Optional quadratic penalty on SAT changes.
    zat_smooth_weight:
        Optional quadratic penalty on ZAT changes.
    previous_action:
        Real action executed at the previous outer step. When provided,
        the first smoothness penalty is measured relative to this action.

    Returns
    -------
    MPCRolloutResult
        Predicted cumulative metrics and per-step records.
    """
    actions = _validate_action_sequence(env, action_sequence)

    if len(actions) == 0:
        raise ValueError("MPC horizon must contain at least one action.")

    sat_smooth_weight = float(sat_smooth_weight)
    zat_smooth_weight = float(zat_smooth_weight)

    if sat_smooth_weight < 0.0 or zat_smooth_weight < 0.0:
        raise ValueError("Smoothness weights must be nonnegative.")

    current_state = initial_state.copy()

    total_cost = 0.0
    total_reward = 0.0
    total_energy_kwh = 0.0
    total_temp_exceed_degC_hr = 0.0
    violation_hours = 0.0

    records: List[MPCStepRecord] = []

    dt_hour = float(env.dt) / 3600.0

    if previous_action is None:
        prev_action = None
    else:
        prev_action = _validate_action(env, previous_action)

    for action in actions:
        current_state, record = propagate_one_step(
            env=env,
            state=current_state,
            action=action,
        )

        step_cost = record.cost

        if prev_action is not None:
            sat_delta = float(action[0] - prev_action[0])
            zat_delta = float(action[1] - prev_action[1])

            smooth_cost = (
                sat_smooth_weight * sat_delta ** 2
                + zat_smooth_weight * zat_delta ** 2
            )

            step_cost += smooth_cost
            record.cost = float(step_cost)

        total_cost += step_cost
        total_reward += record.reward
        total_energy_kwh += record.energy_kwh
        total_temp_exceed_degC_hr += record.temp_exceed_degC * dt_hour

        if record.temp_exceed_degC > 0.0:
            violation_hours += dt_hour

        records.append(record)
        prev_action = action.copy()

    return MPCRolloutResult(
        total_cost=float(total_cost),
        total_reward=float(total_reward),
        total_energy_kwh=float(total_energy_kwh),
        total_temp_exceed_degC_hr=float(total_temp_exceed_degC_hr),
        violation_hours=float(violation_hours),
        final_state=current_state,
        records=records,
    )


def flatten_action_sequence(action_sequence: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert shape (horizon, 2) to the optimizer vector shape (2*horizon,)."""
    actions = np.asarray(action_sequence, dtype=np.float64)

    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("Expected action_sequence with shape (horizon, 2).")

    return actions.reshape(-1)


def unflatten_action_sequence(
    flat_actions: Sequence[float],
    horizon: int,
) -> np.ndarray:
    """Convert optimizer vector shape (2*horizon,) back to (horizon, 2)."""
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")

    flat = np.asarray(flat_actions, dtype=np.float64).reshape(-1)

    expected_size = 2 * horizon
    if flat.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} optimizer values for horizon={horizon}, "
            f"got {flat.size}."
        )

    return flat.reshape(horizon, 2)


def rollout_to_dict(result: MPCRolloutResult) -> Dict[str, Any]:
    """
    Convert a rollout result to plain Python / NumPy containers.

    Useful for debugging, CSV preparation, and controller diagnostics.
    """
    return {
        "total_cost": result.total_cost,
        "total_reward": result.total_reward,
        "total_energy_kwh": result.total_energy_kwh,
        "total_temp_exceed_degC_hr": result.total_temp_exceed_degC_hr,
        "violation_hours": result.violation_hours,
        "final_state": {
            "raw_state": result.final_state.raw_state.copy(),
            "time_hour": result.final_state.time_hour,
            "integral_error": result.final_state.integral_error,
            "prev_zat_sp": result.final_state.prev_zat_sp,
            "damper_signal_prev": result.final_state.damper_signal_prev,
            "m_fan": result.final_state.m_fan,
        },
        "records": [
            {
                "action": record.action.copy(),
                "reward": record.reward,
                "cost": record.cost,
                "energy_kwh": record.energy_kwh,
                "temp_exceed_degC": record.temp_exceed_degC,
                "comfort_lb": record.comfort_lb,
                "comfort_ub": record.comfort_ub,
                "t_env": record.t_env,
                "t_zone": record.t_zone,
                "t_out": record.t_out,
                "qsg": record.qsg,
                "qint": record.qint,
                "hour": record.hour,
                "damper_signal": record.damper_signal,
                "m_fan": record.m_fan,
            }
            for record in result.records
        ],
    }


def _demo():
    """
    Optional manual smoke test.

    Requirements:
      - Env_develop_mpc_draft.py in the same directory.
      - Weather CSV under ./data/.
    """
    from Env_develop_mpc_draft import ContinuousBuildingControlEnvironment

    env = ContinuousBuildingControlEnvironment(
        data_file="weather_data_2013_to_2017_summer_pandas.csv",
        dt=1800.0,
        start=17664.0,
        end=19872.5,
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369,
    )

    env.reset()
    initial_state = mpc_state_from_env(env)

    actions = np.array(
        [
            [14.5, 23.0],
            [14.5, 23.0],
            [14.5, 23.0],
            [14.5, 23.0],
            [14.5, 23.0],
            [14.5, 23.0],
        ],
        dtype=np.float64,
    )

    rollout = rollout_prediction(
        env=env,
        initial_state=initial_state,
        action_sequence=actions,
    )

    print("MPC rollout smoke test")
    print(f"  Horizon: {len(rollout.records)}")
    print(f"  Total cost: {rollout.total_cost:.8f}")
    print(f"  Total reward: {rollout.total_reward:.8f}")
    print(f"  Total energy: {rollout.total_energy_kwh:.8f} kWh")
    print(
        "  Temperature exceedance: "
        f"{rollout.total_temp_exceed_degC_hr:.8f} degC-hr"
    )
    print(f"  Violation hours: {rollout.violation_hours:.4f} h")
    print(f"  Final T_zone: {rollout.final_state.raw_state[1]:.6f} °C")

    # Confirm rollout did not mutate the real environment.
    assert np.isclose(env.t, env.start)
    assert np.allclose(
        env.get_mpc_state()["raw_state"],
        initial_state.raw_state,
    )

    print("  Real environment unchanged: PASS")


if __name__ == "__main__":
    _demo()

# =============================================================================
# Online parameter-learning MPC support
# =============================================================================

from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping

import pandas as pd
from scipy.optimize import least_squares

from parameter_identification_model import (
    PARAMETER_NAMES,
    ThreeRTwoCParameters,
    compute_residual_vector,
)


class OnlineParameterLearningMPC:
    """Manage generic-model MPC, one-time online identification, and model switch."""

    def __init__(
        self,
        prediction_env_factory: Callable[[Mapping[str, float]], Any],
        generic_parameters: Mapping[str, float],
        parameter_bounds: Mapping[str, Tuple[float, float]],
        identification_steps: int = 48,
        envelope_weight: float = 1.0,
        zone_weight: float = 1.0,
        max_nfev: int = 500,
        ftol: float = 1e-10,
        xtol: float = 1e-10,
        gtol: float = 1e-10,
        loss: str = "linear",
        output_directory: Path | str = "results/online_mpc/identification",
        verbose: bool = True,
    ) -> None:
        self.prediction_env_factory = prediction_env_factory
        self.generic_parameters = ThreeRTwoCParameters.from_mapping(generic_parameters)
        self.current_parameters = self.generic_parameters
        self.identified_parameters: Optional[ThreeRTwoCParameters] = None
        self.parameter_bounds = dict(parameter_bounds)
        self.identification_steps = int(identification_steps)
        self.envelope_weight = float(envelope_weight)
        self.zone_weight = float(zone_weight)
        self.max_nfev = int(max_nfev)
        self.ftol = float(ftol)
        self.xtol = float(xtol)
        self.gtol = float(gtol)
        self.loss = str(loss)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.verbose = bool(verbose)

        if self.identification_steps < 1:
            raise ValueError("identification_steps must be positive.")

        self.phase = "generic"
        self.prediction_env = self.prediction_env_factory(
            self.current_parameters.to_dict()
        )
        self.rows: List[Dict[str, float]] = []
        self.collected_steps = 0
        self.identification_result = None
        self.identification_elapsed_seconds: Optional[float] = None

    def synchronize_prediction_env_time(self, real_env) -> None:
        """Keep simple time attributes aligned when they exist."""
        if hasattr(self.prediction_env, "t") and hasattr(real_env, "t"):
            self.prediction_env.t = float(real_env.t)

    @staticmethod
    def _snapshot_value(snapshot: Mapping[str, Any], key: str, default=np.nan) -> float:
        value = snapshot.get(key, default)
        if value is None:
            return float("nan")
        return float(value)

    def _build_row(
        self,
        start_snapshot: Mapping[str, Any],
        action: Sequence[float],
        end_snapshot: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> Dict[str, float]:
        start_raw = np.asarray(start_snapshot["raw_state"], dtype=np.float64)
        end_raw = np.asarray(end_snapshot["raw_state"], dtype=np.float64)
        action_arr = np.asarray(action, dtype=np.float64).reshape(2)

        return {
            "time_hour_start": float(start_snapshot["time_hour"]),
            "T_env_start": float(start_raw[0]),
            "T_zone_start": float(start_raw[1]),
            "SAT_sp": float(action_arr[0]),
            "ZAT_sp": float(action_arr[1]),
            "integral_error_start": self._snapshot_value(start_snapshot, "integral_error", 0.0),
            "prev_zat_sp_start": self._snapshot_value(start_snapshot, "prev_zat_sp", np.nan),
            "damper_signal_prev_start": self._snapshot_value(start_snapshot, "damper_signal_prev", 0.0),
            "T_out_used": float(info.get("T_out_raw", start_raw[3])),
            "T_cor_used": float(start_raw[2]),
            "Qsg_used": float(info.get("Qsg_raw", start_raw[4])),
            "Qint_used": float(info.get("Qint_raw", start_raw[5])),
            "T_env_end": float(end_raw[0]),
            "T_zone_end": float(end_raw[1]),
        }

    def record_transition(
        self,
        start_snapshot: Mapping[str, Any],
        action: Sequence[float],
        end_snapshot: Mapping[str, Any],
        info: Mapping[str, Any],
        step_index: Optional[int] = None,
    ) -> bool:
        """Record one completed real transition; identify once enough rows exist."""
        del step_index
        if self.phase != "generic":
            return False

        self.rows.append(self._build_row(start_snapshot, action, end_snapshot, info))
        self.collected_steps = len(self.rows)

        if self.collected_steps < self.identification_steps:
            return False

        self._identify_and_switch()
        return True

    def _identify_and_switch(self) -> None:
        dataset = pd.DataFrame(self.rows[: self.identification_steps])
        dataset_path = self.output_directory / "online_identification_dataset.csv"
        dataset.to_csv(dataset_path, index=False)

        x0 = self.generic_parameters.to_array()
        lower = np.array(
            [self.parameter_bounds[name][0] for name in PARAMETER_NAMES],
            dtype=np.float64,
        )
        upper = np.array(
            [self.parameter_bounds[name][1] for name in PARAMETER_NAMES],
            dtype=np.float64,
        )

        def residuals(x: np.ndarray) -> np.ndarray:
            return compute_residual_vector(
                parameters=x,
                dataset=dataset,
                envelope_weight=self.envelope_weight,
                zone_weight=self.zone_weight,
                simulation_mode="one_step",
            )

        if self.verbose:
            print("\n" + "=" * 78)
            print("ONLINE MPC PARAMETER IDENTIFICATION")
            print("=" * 78)
            print(f"Rows used: {len(dataset)}")

        start_time = perf_counter()
        result = least_squares(
            residuals,
            x0=x0,
            bounds=(lower, upper),
            max_nfev=self.max_nfev,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            loss=self.loss,
            verbose=0,
        )
        self.identification_elapsed_seconds = perf_counter() - start_time
        self.identification_result = result
        self.identified_parameters = ThreeRTwoCParameters.from_array(result.x)
        self.current_parameters = self.identified_parameters

        old_env = self.prediction_env
        self.prediction_env = self.prediction_env_factory(
            self.current_parameters.to_dict()
        )
        if hasattr(old_env, "close"):
            old_env.close()

        self.phase = "identified"

        pd.DataFrame(
            [
                {
                    "parameter": name,
                    "generic_value": self.generic_parameters.to_dict()[name],
                    "identified_value": self.current_parameters.to_dict()[name],
                }
                for name in PARAMETER_NAMES
            ]
        ).to_csv(
            self.output_directory / "online_identified_parameters.csv",
            index=False,
        )

        if self.verbose:
            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            print(f"Function evaluations: {result.nfev}")
            print(f"Elapsed time: {self.identification_elapsed_seconds:.6f} s")
            for name in PARAMETER_NAMES:
                print(f"  {name:<8} = {self.current_parameters.to_dict()[name]:.12g}")
            print("MPC prediction model switched to identified parameters.")
            print("=" * 78 + "\n")
