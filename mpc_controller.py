from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from mpc_model import (
    MPCState,
    MPCRolloutResult,
    flatten_action_sequence,
    mpc_state_from_env,
    rollout_prediction,
    unflatten_action_sequence,
)


@dataclass
class MPCDiagnostics:
    """Solver information returned after each MPC optimization."""

    success: bool
    message: str
    objective: float
    iterations: int
    function_evaluations: int
    solve_time_sec: float
    used_fallback: bool
    initial_guess: np.ndarray
    optimal_sequence: np.ndarray
    predicted_rollout: Optional[MPCRolloutResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "objective": self.objective,
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "solve_time_sec": self.solve_time_sec,
            "used_fallback": self.used_fallback,
            "initial_guess": self.initial_guess.copy(),
            "optimal_sequence": self.optimal_sequence.copy(),
            "predicted_rollout": self.predicted_rollout,
        }


class MPCController:
    """
    Receding-horizon MPC controller for the SAT/ZAT HVAC environment.

    The controller:
      1. snapshots the current real environment state,
      2. optimizes a future SAT/ZAT sequence,
      3. returns only the first action,
      4. stores the solution for warm-starting the next optimization.

    Parameters
    ----------
    env:
        MPC-compatible environment implementing get_mpc_state() and predict_step().
    horizon:
        Number of outer environment steps in the prediction horizon.
        With dt=1800 s, horizon=6 corresponds to 3 hours.
    method:
        SciPy optimization method. SLSQP is the default.
    maxiter:
        Maximum optimizer iterations per MPC solve.
    ftol:
        Function tolerance passed to SLSQP.
    sat_smooth_weight:
        Optional quadratic penalty on SAT changes.
    zat_smooth_weight:
        Optional quadratic penalty on ZAT changes.
    default_action:
        Initial/fallback [SAT, ZAT].
        If omitted, the midpoint of the action bounds is used.
    """

    def __init__(
        self,
        env,
        horizon: int = 6,
        method: str = "SLSQP",
        maxiter: int = 100,
        ftol: float = 1e-6,
        sat_smooth_weight: float = 0.0,
        zat_smooth_weight: float = 0.0,
        default_action: Optional[Sequence[float]] = None,
    ):
        self.env = env
        self.horizon = int(horizon)
        self.method = str(method)
        self.maxiter = int(maxiter)
        self.ftol = float(ftol)
        self.sat_smooth_weight = float(sat_smooth_weight)
        self.zat_smooth_weight = float(zat_smooth_weight)

        if self.horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive.")
        if self.ftol <= 0.0:
            raise ValueError("ftol must be positive.")
        if self.sat_smooth_weight < 0.0 or self.zat_smooth_weight < 0.0:
            raise ValueError("Smoothness weights must be nonnegative.")

        self.action_low = np.asarray(
            self.env.action_space.low,
            dtype=np.float64,
        )
        self.action_high = np.asarray(
            self.env.action_space.high,
            dtype=np.float64,
        )

        if self.action_low.shape != (2,) or self.action_high.shape != (2,):
            raise ValueError(
                "This controller expects a 2-D action space [SAT, ZAT]."
            )

        if default_action is None:
            self.default_action = 0.5 * (
                self.action_low + self.action_high
            )
        else:
            self.default_action = np.clip(
                np.asarray(default_action, dtype=np.float64).reshape(-1),
                self.action_low,
                self.action_high,
            )

        if self.default_action.size != 2:
            raise ValueError("default_action must contain [SAT, ZAT].")

        self.previous_solution: Optional[np.ndarray] = None
        self.previous_executed_action: Optional[np.ndarray] = None

        self.bounds = []
        for _ in range(self.horizon):
            self.bounds.append(
                (float(self.action_low[0]), float(self.action_high[0]))
            )
            self.bounds.append(
                (float(self.action_low[1]), float(self.action_high[1]))
            )

    def reset(self) -> None:
        """Clear warm-start memory when a new real episode begins."""
        self.previous_solution = None
        self.previous_executed_action = None

    def set_previous_action(self, action: Sequence[float]) -> None:
        """
        Store the most recently executed real action.

        This is only needed when using nonzero smoothness weights.
        """
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_arr.size != 2:
            raise ValueError("action must contain [SAT, ZAT].")

        self.previous_executed_action = np.clip(
            action_arr,
            self.action_low,
            self.action_high,
        )

    def _make_initial_guess(self) -> np.ndarray:
        """
        Generate optimizer initial guess.

        First solve:
            repeat default_action across the horizon.

        Later solves:
            shift the previous optimal sequence left by one step and repeat
            its final action at the end.
        """
        if self.previous_solution is None:
            initial_sequence = np.tile(
                self.default_action,
                (self.horizon, 1),
            )
        else:
            previous_sequence = np.asarray(
                self.previous_solution,
                dtype=np.float64,
            ).reshape(self.horizon, 2)

            initial_sequence = np.vstack(
                [
                    previous_sequence[1:],
                    previous_sequence[-1],
                ]
            )

        initial_sequence = np.clip(
            initial_sequence,
            self.action_low,
            self.action_high,
        )

        return flatten_action_sequence(initial_sequence)

    def _objective(
        self,
        flat_actions: np.ndarray,
        initial_state: MPCState,
    ) -> float:
        """
        Objective passed to scipy.optimize.minimize.

        The environment remains unchanged because rollout_prediction()
        uses env.predict_step().
        """
        actions = unflatten_action_sequence(
            flat_actions,
            horizon=self.horizon,
        )

        rollout = rollout_prediction(
            env=self.env,
            initial_state=initial_state,
            action_sequence=actions,
            sat_smooth_weight=self.sat_smooth_weight,
            zat_smooth_weight=self.zat_smooth_weight,
            previous_action=self.previous_executed_action,
        )

        total_cost = float(rollout.total_cost)

        if not np.isfinite(total_cost):
            return 1e12

        return total_cost

    def _build_fallback_sequence(self, x0: np.ndarray) -> np.ndarray:
        """
        Construct a safe sequence if the optimizer fails.

        Preference order:
          1. shifted warm-start sequence already contained in x0,
          2. repeated default action.
        """
        try:
            sequence = unflatten_action_sequence(
                x0,
                horizon=self.horizon,
            )
        except (TypeError, ValueError):
            sequence = np.tile(
                self.default_action,
                (self.horizon, 1),
            )

        return np.clip(
            sequence,
            self.action_low,
            self.action_high,
        )

    def solve(
        self,
        initial_state: Optional[MPCState] = None,
    ) -> Tuple[np.ndarray, MPCDiagnostics]:
        """
        Solve one receding-horizon MPC problem.

        Parameters
        ----------
        initial_state:
            Optional explicit MPCState. If omitted, the current real
            environment state is read using mpc_state_from_env(env).

        Returns
        -------
        best_action:
            First [SAT, ZAT] action of the optimized sequence.
        diagnostics:
            MPCDiagnostics with solver and predicted rollout information.
        """
        if initial_state is None:
            initial_state = mpc_state_from_env(self.env)
        else:
            initial_state = initial_state.copy()

        x0 = self._make_initial_guess()

        start_time = perf_counter()

        try:
            result: OptimizeResult = minimize(
                fun=self._objective,
                x0=x0,
                args=(initial_state,),
                method=self.method,
                bounds=self.bounds,
                options={
                    "maxiter": self.maxiter,
                    "ftol": self.ftol,
                    "eps": 1e-3,
                    "disp": False,
                },
            )

            elapsed = perf_counter() - start_time

            optimizer_success = bool(
                result.success
                and result.x is not None
                and np.all(np.isfinite(result.x))
                and np.isfinite(result.fun)
            )

            if optimizer_success:
                optimal_sequence = unflatten_action_sequence(
                    result.x,
                    horizon=self.horizon,
                )
                used_fallback = False
                message = str(result.message)
            else:
                optimal_sequence = self._build_fallback_sequence(x0)
                used_fallback = True
                message = (
                    f"Optimizer failed: {getattr(result, 'message', 'unknown')}. "
                    "Fallback sequence used."
                )

            optimal_sequence = np.clip(
                optimal_sequence,
                self.action_low,
                self.action_high,
            )

            predicted_rollout = rollout_prediction(
                env=self.env,
                initial_state=initial_state,
                action_sequence=optimal_sequence,
                sat_smooth_weight=self.sat_smooth_weight,
                zat_smooth_weight=self.zat_smooth_weight,
                previous_action=self.previous_executed_action,
            )

            objective_value = float(predicted_rollout.total_cost)

            self.previous_solution = optimal_sequence.copy()

            best_action = optimal_sequence[0].copy()

            diagnostics = MPCDiagnostics(
                success=optimizer_success,
                message=message,
                objective=objective_value,
                iterations=int(getattr(result, "nit", 0) or 0),
                function_evaluations=int(
                    getattr(result, "nfev", 0) or 0
                ),
                solve_time_sec=float(elapsed),
                used_fallback=used_fallback,
                initial_guess=unflatten_action_sequence(
                    x0,
                    horizon=self.horizon,
                ).copy(),
                optimal_sequence=optimal_sequence.copy(),
                predicted_rollout=predicted_rollout,
            )

            return best_action, diagnostics

        except Exception as exc:
            elapsed = perf_counter() - start_time

            fallback_sequence = self._build_fallback_sequence(x0)

            predicted_rollout: Optional[MPCRolloutResult]
            objective_value: float

            try:
                predicted_rollout = rollout_prediction(
                    env=self.env,
                    initial_state=initial_state,
                    action_sequence=fallback_sequence,
                    sat_smooth_weight=self.sat_smooth_weight,
                    zat_smooth_weight=self.zat_smooth_weight,
                    previous_action=self.previous_executed_action,
                )
                objective_value = float(predicted_rollout.total_cost)
            except Exception:
                predicted_rollout = None
                objective_value = float("nan")

            self.previous_solution = fallback_sequence.copy()

            best_action = fallback_sequence[0].copy()

            diagnostics = MPCDiagnostics(
                success=False,
                message=f"Solver exception: {type(exc).__name__}: {exc}",
                objective=objective_value,
                iterations=0,
                function_evaluations=0,
                solve_time_sec=float(elapsed),
                used_fallback=True,
                initial_guess=unflatten_action_sequence(
                    x0,
                    horizon=self.horizon,
                ).copy(),
                optimal_sequence=fallback_sequence.copy(),
                predicted_rollout=predicted_rollout,
            )

            return best_action, diagnostics


def _demo():
    """
    Optional one-solve smoke test.

    Requirements:
      - Env_develop_mpc_draft.py in the same directory.
      - mpc_model.py in the same directory.
      - Weather CSV under ./data/.
    """
    from Env_develop_mpc_draft import (
        ContinuousBuildingControlEnvironment,
    )

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

    controller = MPCController(
        env=env,
        horizon=6,
        method="SLSQP",
        maxiter=100,
        ftol=1e-6,
        sat_smooth_weight=0.0,
        zat_smooth_weight=0.0,
        default_action=[14.5, 23.0],
    )

    initial_time = env.t
    initial_state = env.state.copy()
    initial_integral = env.integral_error
    initial_damper = env.damper_signal_prev

    action, diagnostics = controller.solve()

    print("MPC controller one-solve smoke test")
    print(f"  Success: {diagnostics.success}")
    print(f"  Message: {diagnostics.message}")
    print(f"  Best first action [SAT, ZAT]: {action}")
    print(f"  Objective: {diagnostics.objective:.10f}")
    print(f"  Iterations: {diagnostics.iterations}")
    print(
        f"  Function evaluations: "
        f"{diagnostics.function_evaluations}"
    )
    print(f"  Solve time: {diagnostics.solve_time_sec:.6f} s")
    print(f"  Used fallback: {diagnostics.used_fallback}")

    if diagnostics.predicted_rollout is not None:
        rollout = diagnostics.predicted_rollout
        print(
            f"  Predicted energy over horizon: "
            f"{rollout.total_energy_kwh:.8f} kWh"
        )
        print(
            f"  Predicted exceedance over horizon: "
            f"{rollout.total_temp_exceed_degC_hr:.8f} degC-hr"
        )
        print(
            f"  Predicted final T_zone: "
            f"{rollout.final_state.raw_state[1]:.6f} °C"
        )

    assert np.isclose(env.t, initial_time)
    assert np.allclose(env.state, initial_state)
    assert np.isclose(env.integral_error, initial_integral)
    assert np.isclose(env.damper_signal_prev, initial_damper)

    print("  Real environment unchanged: PASS")

    # Demonstrate the receding-horizon execution pattern.
    obs, reward, done, info = env.step(action)
    controller.set_previous_action(action)

    print("  Executed first MPC action in real environment")
    print(f"  Real reward: {reward:.10f}")
    print(f"  Real T_zone: {info['T_zone_raw']:.6f} °C")
    print(f"  Real energy: {info['TotalEnergy_kWh']:.8f} kWh")
    print(f"  Done: {done}")


if __name__ == "__main__":
    _demo()