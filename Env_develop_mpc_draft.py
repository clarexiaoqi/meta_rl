__all__ = ["ContinuousBuildingControlEnvironment"]

import numpy as np
from scipy import signal
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding

DATA_PATH = "./data/"


class ContinuousBuildingControlEnvironment(gym.Env):
    """
    2-action single-setpoint control (SCALED reward):
      - Action a_t = [SAT_sp, ZAT_sp]
      - Mode (for logging only):
          heating if T_zone < ZAT_sp - mode_deadband
          cooling if T_zone > ZAT_sp + mode_deadband
          neutral otherwise
      - PI loop tracks ZAT_sp always (controls damper / airflow)
      - Terminal reheat is NOT blocked by mode:
          if (damper at min flow) and (T_zone < ZAT_sp) -> reheat can activate
      - Reward (SCALED):
          energy_norm = TotalEnergy_kWh / E_ref
          temp_norm   = TempExceed_degC / T_ref
          reward = -(energy_norm + alpha * temp_norm)
        TempExceed is linear exceed outside day/night comfort bounds:
          day   (7~20): [lb_set, ub_set]
          night (otherwise): [night_lb_set, night_ub_set]
      - damper_signal carries over across outer steps (smoother PI behavior)

    MPC-compatible refactor:
      - _transition(...) contains the shared physical/control logic.
      - step(...) calls _transition(...) and commits the returned state.
      - predict_step(...) calls the same _transition(...) without mutating env.
      - get_mpc_state(...) exposes the current thermal + PI controller state.

    Important:
      - The physical equations, PI controller, energy model, reward, and
        external step() interface are kept unchanged.
    """

    def __init__(
        self,
        data_file,
        dt=1800.0,
        start=0.0,
        end=720.0,
        C_env=None,
        C_air=None,
        R_rc=None,
        R_oe=None,
        R_er=None,
        lb_set=22.0,
        ub_set=24.0,
        night_lb_set=15.0,
        night_ub_set=28.0,
        mode_deadband=0.0,
        alpha=0.3,
        E_ref=60.0,
        T_ref=8.0,
        SAT_low=12.8,
        SAT_high=17.7,
        ZAT_low=18.0,
        ZAT_high=26.0,
        Qh_reheat_max=300.0,
    ):
        self.dt = float(dt)
        self.data = pd.read_csv(DATA_PATH + data_file)
        self.start = float(start)
        self.end = float(end)

        if C_env is None or C_air is None or R_rc is None or R_oe is None or R_er is None:
            raise ValueError("C_env, C_air, R_rc, R_oe, R_er must be provided (not None).")

        self.C_env = float(C_env)
        self.C_air = float(C_air)
        self.R_rc = float(R_rc)
        self.R_oe = float(R_oe)
        self.R_er = float(R_er)
        self.a_sol_env = 0.3

        # PI controller
        self.Kp = 15.0
        self.Ki = 0.02
        self.integral_error = 0.0
        self.prev_ZAT_sp = None
        self.pi_interval = 60.0 * 5.0
        self.m_fan = None
        self.damper_signal_prev = 0.0

        self.mode_deadband = float(mode_deadband)

        # Reheat
        self.Qh_reheat_max = float(Qh_reheat_max)
        self.eta_reheat = 0.9

        # 3R2C continuous model
        A = np.zeros((2, 2))
        B = np.zeros((2, 5))

        A[0, 0] = (-1.0 / self.C_env) * (1.0 / self.R_er + 1.0 / self.R_oe)
        A[0, 1] = 1.0 / (self.C_env * self.R_er)

        A[1, 0] = 1.0 / (self.C_air * self.R_er)
        A[1, 1] = (-1.0 / self.C_air) * (1.0 / self.R_er + 1.0 / self.R_rc)

        B[0, 1] = 1.0 / (self.C_env * self.R_oe)
        B[0, 2] = self.a_sol_env / self.C_env

        B[1, 0] = 1.0 / (self.C_air * self.R_rc)
        B[1, 2] = (1.0 - self.a_sol_env) / self.C_air
        B[1, 3] = 1.0 / self.C_air
        B[1, 4] = 1.0 / self.C_air

        self.Ac, self.Bc = A, B

        disc_dt = signal.StateSpace(
            self.Ac,
            self.Bc,
            np.array([[1.0, 0.0]]),
            np.zeros(5),
        ).to_discrete(dt=self.dt)
        self.A_dt, self.B_dt = disc_dt.A, disc_dt.B

        disc_pi = signal.StateSpace(
            self.Ac,
            self.Bc,
            np.array([[1.0, 0.0]]),
            np.zeros(5),
        ).to_discrete(dt=self.pi_interval)
        self.A_pi, self.B_pi = disc_pi.A, disc_pi.B

        self.n_pi_loops = int(self.dt // self.pi_interval)
        if self.n_pi_loops < 1:
            self.n_pi_loops = 1
        self.dt_hr_pi = self.pi_interval / 3600.0

        # Comfort bounds
        self.day_lb = float(lb_set)
        self.day_ub = float(ub_set)
        self.night_lb = float(night_lb_set)
        self.night_ub = float(night_ub_set)

        # Backward-compatible names
        self.lb = self.day_lb
        self.ub = self.day_ub

        # Reward
        self.alpha = float(alpha)
        self.E_ref = float(E_ref)
        self.T_ref = float(T_ref)

        # York Affinity DNZ060 cooling curves
        self.capft = {
            "C1": 1.2343140,
            "C2": -0.0398816,
            "C3": 0.0019354,
            "C4": 0.0062114,
            "C5": -0.0001247,
            "C6": -0.0003619,
        }

        self.eirft = {
            "C1": -0.1272387,
            "C2": 0.0848124,
            "C3": -0.0021062,
            "C4": -0.0085792,
            "C5": 0.0007783,
            "C6": -0.0005585,
        }

        self.capfff = {"C1": 1.2527302, "C2": -0.7182445, "C3": 0.4623738}
        self.eirfff = {"C1": 0.6529892, "C2": 0.8193151, "C3": -0.4617716}
        self.COP_rated = 4.24

        # HVAC constants
        self.cp_air = 1004.0
        self.m_dot_min = 0.080939
        self.m_design = 0.9264 * 0.4
        self.m_dot_max = self.m_dot_min * 550.0 / 140.0

        self.dP = 500.0
        self.e_tot = 0.6045
        self.rho_air = 1.225
        self.c_FAN = np.array([0.04076, 0.08804, -0.07293, 0.94374, 0.0])

        self.capacity_scale = 1.0 / 3.0

        # Action / observation spaces
        self.action_space = spaces.Box(
            low=np.array([SAT_low, ZAT_low], dtype=np.float32),
            high=np.array([SAT_high, ZAT_high], dtype=np.float32),
            dtype=np.float32,
        )

        # State = [T_env, T_zone, T_cor, T_out, Qsg, Qint, hour_sin, hour_cos]
        self.low = np.array([10.0, 15.0, 20.0, -40.0, 0.0, 50.0, -1.0, -1.0], dtype=np.float32)
        self.high = np.array([45.0, 28.0, 28.0, 40.0, 1100.0, 180.0, 1.0, 1.0], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )

        self.state = None
        self.t = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _hour_to_cyclic(self, hour):
        hour_rad = 2.0 * np.pi * (float(hour) % 24.0) / 24.0
        return np.sin(hour_rad), np.cos(hour_rad)

    def _apply_action_setpoints(self, a_t):
        a_t = np.asarray(a_t, dtype=np.float64).reshape(-1)

        if a_t.size != 2:
            raise ValueError(
                f"Expected action [SAT, ZAT], got shape {a_t.shape}."
            )

        SAT_sp = float(
            np.clip(
                a_t[0],
                float(self.action_space.low[0]),
                float(self.action_space.high[0]),
            )
        )

        ZAT_sp = float(
            np.clip(
                a_t[1],
                float(self.action_space.low[1]),
                float(self.action_space.high[1]),
            )
        )

        return SAT_sp, ZAT_sp
        if a_t.size != 2:
            raise ValueError(f"Expected action with 2 values [SAT, ZAT], got shape {a_t.shape}.")

        SAT_sp = float(np.clip(a_t[0], self.action_space.low[0], self.action_space.high[0]))
        ZAT_sp = float(np.clip(a_t[1], self.action_space.low[1], self.action_space.high[1]))
        return SAT_sp, ZAT_sp

    def _select_mode(self, T_zone, ZAT_sp):
        if T_zone < ZAT_sp - self.mode_deadband:
            return "heating"
        if T_zone > ZAT_sp + self.mode_deadband:
            return "cooling"
        return "neutral"

    def _get_raw_state(self):
        if self.state is None:
            raise RuntimeError("Environment state is None. Call reset() before step() or get_mpc_state().")
        return self.state * (self.high - self.low) + self.low

    def _get_disturbance_at_time(self, time_hour):
        """
        Read exogenous inputs at the specified absolute simulation time.

        The current dataset is sampled every 30 minutes, so the indexing
        intentionally preserves the original int(time_hour * 2) behavior.
        """
        idx = min(max(int(float(time_hour) * 2), 0), len(self.data) - 1)
        row = self.data.iloc[idx]

        return {
            "T_out": float(row.Tout),
            "Qsg": float(row.Qsg),
            "Qint": float(row.Qint),
            "Hour": float(row.Hour),
            "T_cor": 24.0,
        }

    def get_mpc_state(self):
        """
        Return a snapshot of the current thermal and PI-controller state.

        The returned arrays/values are copies, so modifying this dictionary
        does not mutate the real environment.
        """
        raw_state = self._get_raw_state()

        return {
            "raw_state": raw_state.copy(),
            "time_hour": float(self.t),
            "integral_error": float(self.integral_error),
            "prev_zat_sp": None if self.prev_ZAT_sp is None else float(self.prev_ZAT_sp),
            "damper_signal_prev": float(self.damper_signal_prev),
            "m_fan": float(self.m_fan if self.m_fan is not None else self.m_dot_min),
        }

    def _transition(
        self,
        raw_state,
        action,
        time_hour,
        integral_error,
        prev_zat_sp,
        damper_signal_prev,
    ):
        """
        Shared one-step building transition.

        This function is side-effect free: it does not modify self.state,
        self.t, self.integral_error, self.prev_ZAT_sp, self.m_fan, or
        self.damper_signal_prev.

        Parameters
        ----------
        raw_state : array-like, shape (8,)
            Denormalized state:
            [T_env, T_zone, T_cor, T_out, Qsg, Qint, hour_sin, hour_cos].
        action : array-like, shape (2,)
            [SAT_sp, ZAT_sp].
        time_hour : float
            Current absolute simulation time in hours.
        integral_error : float
            Current PI integral state.
        prev_zat_sp : float or None
            Previously used ZAT setpoint.
        damper_signal_prev : float
            Damper signal carried from the previous outer step.

        Returns
        -------
        dict
            Next state, next PI states, reward, done, and info.
        """
        raw_state = np.asarray(raw_state, dtype=np.float64).reshape(-1)
        if raw_state.size != 8:
            raise ValueError(f"Expected raw_state with 8 values, got shape {raw_state.shape}.")

        T_zone = float(raw_state[1])

        # Preserve original timing: advance first, then read exogenous inputs.
        next_time = float(time_hour) + self.dt / 3600.0
        disturbance = self._get_disturbance_at_time(next_time)

        T_out = disturbance["T_out"]
        Qsg = disturbance["Qsg"]
        Qint = disturbance["Qint"]
        Hour = disturbance["Hour"]
        T_cor = disturbance["T_cor"]

        hour_sin, hour_cos = self._hour_to_cyclic(Hour)

        SAT_sp, ZAT_sp = self._apply_action_setpoints(action)
        mode = self._select_mode(T_zone, ZAT_sp)

        next_integral_error = float(integral_error)
        if prev_zat_sp is None or abs(ZAT_sp - float(prev_zat_sp)) > 0.5:
            next_integral_error = 0.0
        next_prev_zat_sp = float(ZAT_sp)

        x_room = raw_state[:2].copy()

        total_energy = 0.0
        cool_energy_total = 0.0
        heat_energy_total = 0.0
        reheat_energy_total = 0.0
        fan_energy_total = 0.0

        damper_signal = float(damper_signal_prev)
        COPc = self.COP_rated
        m_fan_last = self.m_dot_min

        reheat_signal_last = 0.0
        Q_reheat_last_W = 0.0

        u_base = np.array([T_cor, T_out, Qsg, Qint], dtype=np.float64)

        for _ in range(self.n_pi_loops):
            m_fan = self.m_dot_min + (damper_signal / 100.0) * (self.m_dot_max - self.m_dot_min)

            error = T_zone - ZAT_sp
            raw = self.Kp * error + self.Ki * next_integral_error
            damper_signal = float(np.clip(raw, 0.0, 100.0))

            if not ((raw >= 99.9 and error > 0) or (raw <= 0.1 and error < 0)):
                next_integral_error += error * self.pi_interval

            m_fan = self.m_dot_min + (damper_signal / 100.0) * (self.m_dot_max - self.m_dot_min)
            m_fan_last = float(m_fan)

            Q_air = self.capacity_scale * (m_fan * self.cp_air * (SAT_sp - T_zone))

            at_min_flow = damper_signal <= 1.0
            if at_min_flow and (T_zone < ZAT_sp):
                reheat_signal = float(np.clip((ZAT_sp - T_zone) / 3.0, 0.0, 1.0))
                Q_reheat = reheat_signal * self.Qh_reheat_max
            else:
                reheat_signal = 0.0
                Q_reheat = 0.0

            reheat_signal_last = float(reheat_signal)
            Q_reheat_last_W = float(Q_reheat)

            u_total = Q_air + Q_reheat
            u_model = np.array(
                [u_base[0], u_base[1], u_base[2], u_base[3], u_total],
                dtype=np.float64,
            )

            x_room = self.A_pi @ x_room + self.B_pi @ u_model
            T_zone = float(x_room[1])

            f_flow = max(0.05, m_fan / self.m_design)

            capfff = (
                self.capfff["C1"]
                + self.capfff["C2"] * f_flow
                + self.capfff["C3"] * f_flow ** 2
            )
            eirfff = (
                self.eirfff["C1"]
                + self.eirfff["C2"] * f_flow
                + self.eirfff["C3"] * f_flow ** 2
            )

            capft = (
                self.capft["C1"]
                + self.capft["C2"] * T_out
                + self.capft["C3"] * T_out ** 2
                + self.capft["C4"] * SAT_sp
                + self.capft["C5"] * SAT_sp ** 2
                + self.capft["C6"] * T_out * SAT_sp
            )

            eirft = (
                self.eirft["C1"]
                + self.eirft["C2"] * T_out
                + self.eirft["C3"] * T_out ** 2
                + self.eirft["C4"] * SAT_sp
                + self.eirft["C5"] * SAT_sp ** 2
                + self.eirft["C6"] * T_out * SAT_sp
            )

            COPc = max(0.1, (self.COP_rated * capft * capfff) / (eirft * eirfff))

            f_pl = (
                self.c_FAN[0]
                + self.c_FAN[1] * f_flow
                + self.c_FAN[2] * f_flow ** 2
                + self.c_FAN[3] * f_flow ** 3
            )
            Q_fan = f_pl * self.m_design * self.dP / (self.e_tot * self.rho_air)

            P_cool = max(-Q_air, 0.0) / COPc
            P_heat = max(Q_air, 0.0) / self.eta_reheat
            P_reheat = Q_reheat / self.eta_reheat

            cool_step = (P_cool / 1000.0) * self.dt_hr_pi
            heat_step = (P_heat / 1000.0) * self.dt_hr_pi
            reheat_step = (P_reheat / 1000.0) * self.dt_hr_pi
            fan_step = (Q_fan / 1000.0) * self.dt_hr_pi

            total_energy += cool_step + heat_step + reheat_step + fan_step
            cool_energy_total += cool_step
            heat_energy_total += heat_step
            reheat_energy_total += reheat_step
            fan_energy_total += fan_step

        T_now = float(x_room[1])
        if 7 <= Hour <= 20:
            comfort_lb = self.day_lb
            comfort_ub = self.day_ub
            comfort_period = "day"
        else:
            comfort_lb = self.night_lb
            comfort_ub = self.night_ub
            comfort_period = "night"

        Temp_exceed = 0.0
        if T_now < comfort_lb:
            Temp_exceed = comfort_lb - T_now
        elif T_now > comfort_ub:
            Temp_exceed = T_now - comfort_ub

        energy_norm = total_energy / max(self.E_ref, 1e-6)
        temp_norm = Temp_exceed / max(self.T_ref, 1e-6)
        reward = -(energy_norm + self.alpha * temp_norm)

        s_ext = np.array(
            [T_cor, T_out, Qsg, Qint, hour_sin, hour_cos],
            dtype=np.float64,
        )
        next_raw_state = np.concatenate([x_room, s_ext])
        next_normalized_state = (next_raw_state - self.low) / (self.high - self.low)

        done = next_time >= self.end

        T_env_norm = float(next_normalized_state[0])
        T_zone_norm = float(next_normalized_state[1])
        T_cor_norm = float(next_normalized_state[2])
        T_out_norm = float(next_normalized_state[3])
        Qsg_norm = float(next_normalized_state[4])
        Qint_norm = float(next_normalized_state[5])
        hour_sin_norm = float(next_normalized_state[6])
        hour_cos_norm = float(next_normalized_state[7])

        info = {
            "Mode": mode,
            "SAT_sp": float(SAT_sp),
            "ZAT_sp_used": float(ZAT_sp),

            "m_fan": float(m_fan_last),
            "DamperSignal": float(damper_signal),

            "ReheatSignal": float(reheat_signal_last),
            "ReheatPct": float(100.0 * reheat_signal_last),
            "Q_reheat_W": float(Q_reheat_last_W),

            "TotalEnergy_kWh": float(total_energy),
            "EnergyNorm": float(energy_norm),
            "CoolingEnergy_kWh": float(cool_energy_total),
            "HeatingEnergy_kWh": float(heat_energy_total),
            "ReheatEnergy_kWh": float(reheat_energy_total),
            "FanEnergy_kWh": float(fan_energy_total),

            "Hour": float(Hour),
            "hour_sin": float(hour_sin),
            "hour_cos": float(hour_cos),
            "ComfortPeriod": comfort_period,
            "ComfortLB": float(comfort_lb),
            "ComfortUB": float(comfort_ub),

            "lb": float(comfort_lb),
            "ub": float(comfort_ub),

            "TempExceed_degC": float(Temp_exceed),
            "TempNorm": float(temp_norm),
            "Reward": float(reward),

            "T_env_raw": float(x_room[0]),
            "T_zone_raw": float(x_room[1]),
            "T_cor_raw": float(T_cor),
            "T_out_raw": float(T_out),
            "Qsg_raw": float(Qsg),
            "Qint_raw": float(Qint),

            "T_env_norm": T_env_norm,
            "T_zone_norm": T_zone_norm,
            "T_cor_norm": T_cor_norm,
            "T_out_norm": T_out_norm,
            "Qsg_norm": Qsg_norm,
            "Qint_norm": Qint_norm,
            "hour_sin_norm": hour_sin_norm,
            "hour_cos_norm": hour_cos_norm,

            "T_env_norm_out_of_range": float((T_env_norm < 0.0) or (T_env_norm > 1.0)),
            "T_zone_norm_out_of_range": float((T_zone_norm < 0.0) or (T_zone_norm > 1.0)),
            "T_cor_norm_out_of_range": float((T_cor_norm < 0.0) or (T_cor_norm > 1.0)),
            "T_out_norm_out_of_range": float((T_out_norm < 0.0) or (T_out_norm > 1.0)),
            "Qsg_norm_out_of_range": float((Qsg_norm < 0.0) or (Qsg_norm > 1.0)),
            "Qint_norm_out_of_range": float((Qint_norm < 0.0) or (Qint_norm > 1.0)),
        }

        return {
            "next_time": float(next_time),
            "next_raw_state": next_raw_state.astype(np.float64, copy=True),
            "next_normalized_state": next_normalized_state.astype(np.float32, copy=True),
            "next_integral_error": float(next_integral_error),
            "next_prev_zat_sp": float(next_prev_zat_sp),
            "next_damper_signal": float(damper_signal),
            "next_m_fan": float(m_fan_last),
            "reward": float(reward),
            "done": bool(done),
            "info": info,
        }

    def predict_step(
        self,
        action,
        raw_state=None,
        time_hour=None,
        integral_error=None,
        prev_zat_sp=None,
        damper_signal_prev=None,
    ):
        """
        Predict one outer environment step without mutating the real env.

        Any omitted state argument defaults to the current real environment
        value. For multi-step MPC rollout, pass the returned next values into
        the next predict_step(...) call.
        """
        if raw_state is None:
            raw_state = self._get_raw_state()
        if time_hour is None:
            time_hour = self.t
        if integral_error is None:
            integral_error = self.integral_error
        if prev_zat_sp is None:
            prev_zat_sp = self.prev_ZAT_sp
        if damper_signal_prev is None:
            damper_signal_prev = self.damper_signal_prev

        return self._transition(
            raw_state=raw_state,
            action=action,
            time_hour=time_hour,
            integral_error=integral_error,
            prev_zat_sp=prev_zat_sp,
            damper_signal_prev=damper_signal_prev,
        )

    def step(self, a_t):
        raw_state = self._get_raw_state()

        result = self._transition(
            raw_state=raw_state,
            action=a_t,
            time_hour=self.t,
            integral_error=self.integral_error,
            prev_zat_sp=self.prev_ZAT_sp,
            damper_signal_prev=self.damper_signal_prev,
        )

        # Commit the transition to the real environment.
        self.t = result["next_time"]
        self.state = result["next_normalized_state"].copy()
        self.integral_error = result["next_integral_error"]
        self.prev_ZAT_sp = result["next_prev_zat_sp"]
        self.damper_signal_prev = result["next_damper_signal"]
        self.m_fan = result["next_m_fan"]

        if (self.state[0] < 0.0) or (self.state[0] > 1.0):
            print(
                f"[WARN] T_env out of normalization range at t={self.t:.2f} hr | "
                f"T_env_raw={result['info']['T_env_raw']:.3f}, "
                f"T_env_norm={self.state[0]:.3f}, "
                f"low={self.low[0]:.3f}, high={self.high[0]:.3f}"
            )

        return (
            np.array(self.state, dtype=np.float32),
            result["reward"],
            result["done"],
            result["info"],
        )

    def reset(self):
        self.t = self.start
        self.integral_error = 0.0
        self.prev_ZAT_sp = None
        self.m_fan = self.m_dot_min
        self.damper_signal_prev = 0.0

        T_env_0 = 20.0
        T_zone_0 = 24.0
        T_cor = 24.0

        disturbance = self._get_disturbance_at_time(self.start)
        T_out = disturbance["T_out"]
        Qsg = disturbance["Qsg"]
        Qint = disturbance["Qint"]
        Hour = disturbance["Hour"]

        hour_sin, hour_cos = self._hour_to_cyclic(Hour)

        raw_state = np.array(
            [T_env_0, T_zone_0, T_cor, T_out, Qsg, Qint, hour_sin, hour_cos],
            dtype=np.float32,
        )

        self.state = (raw_state - self.low) / (self.high - self.low)
        return np.array(self.state, dtype=np.float32)


def _single_step_equivalence_demo():
    """
    Optional manual check.

    Run this file directly after placing the weather CSV under ./data/.
    It verifies that predict_step(...) and step(...) produce the same
    one-step result from the same environment state.
    """
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
    action = np.array([14.5, 23.0], dtype=np.float32)

    snapshot = env.get_mpc_state()
    prediction = env.predict_step(
        action=action,
        raw_state=snapshot["raw_state"],
        time_hour=snapshot["time_hour"],
        integral_error=snapshot["integral_error"],
        prev_zat_sp=snapshot["prev_zat_sp"],
        damper_signal_prev=snapshot["damper_signal_prev"],
    )

    obs_real, reward_real, done_real, info_real = env.step(action)

    checks = {
        "normalized_state": np.allclose(
            prediction["next_normalized_state"],
            obs_real,
            atol=1e-6,
            rtol=1e-6,
        ),
        "reward": np.isclose(prediction["reward"], reward_real, atol=1e-9, rtol=1e-9),
        "done": prediction["done"] == done_real,
        "T_zone_raw": np.isclose(
            prediction["info"]["T_zone_raw"],
            info_real["T_zone_raw"],
            atol=1e-8,
            rtol=1e-8,
        ),
        "energy": np.isclose(
            prediction["info"]["TotalEnergy_kWh"],
            info_real["TotalEnergy_kWh"],
            atol=1e-10,
            rtol=1e-8,
        ),
        "damper": np.isclose(
            prediction["next_damper_signal"],
            env.damper_signal_prev,
            atol=1e-10,
            rtol=1e-8,
        ),
        "integral_error": np.isclose(
            prediction["next_integral_error"],
            env.integral_error,
            atol=1e-8,
            rtol=1e-8,
        ),
    }

    print("Single-step prediction/step equivalence:")
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise AssertionError("predict_step(...) and step(...) are not equivalent.")

    print("All checks passed.")


if __name__ == "__main__":
    _single_step_equivalence_demo()