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
    2-action single-setpoint control with daytime/nighttime comfort bounds.

    Action:
      a_t = [SAT_sp, ZAT_sp]

    Comfort:
      Daytime 7~20:
        bound = [lb_set, ub_set]
      Nighttime:
        bound = [15, 28]

    Reward:
      reward = -(EnergyNorm + alpha * TempNorm)

    Here TempNorm is computed from linear temperature exceedance.
    Both daytime and nighttime have comfort penalty, but nighttime bound is wider.
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
        mode_deadband=0.0,
        alpha=3,
        E_ref=60,
        T_ref=8,
        SAT_low=10.0,
        SAT_high=15.5,
        ZAT_low=18.0,
        ZAT_high=26.0,
        Qh_reheat_max=300.0,
    ):
        self.dt = float(dt)
        self.data = pd.read_csv(DATA_PATH + data_file)
        self.start = float(start)
        self.end = float(end)

        if C_env is None or C_air is None or R_rc is None or R_oe is None or R_er is None:
            raise ValueError("C_env, C_air, R_rc, R_oe, R_er must be provided.")

        self.C_env = float(C_env)
        self.C_air = float(C_air)
        self.R_rc = float(R_rc)
        self.R_oe = float(R_oe)
        self.R_er = float(R_er)
        self.a_sol_env = 0.90303

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

        # 3R2C model
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

        # Daytime comfort bounds
        self.lb = float(lb_set)
        self.ub = float(ub_set)

        # Nighttime comfort bounds from old environment
        self.night_lb = 15.0
        self.night_ub = 28.0

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

        self.capfff = {
            "C1": 1.2527302,
            "C2": -0.7182445,
            "C3": 0.4623738,
        }

        self.eirfff = {
            "C1": 0.6529892,
            "C2": 0.8193151,
            "C3": -0.4617716,
        }

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

        # Action space
        self.action_space = spaces.Box(
            low=np.array([SAT_low, ZAT_low], dtype=np.float32),
            high=np.array([SAT_high, ZAT_high], dtype=np.float32),
            dtype=np.float32,
        )

        # State = [T_env, T_zone, T_cor, T_out, Qsg, Qint, Hour]
        self.low = np.array([10., 15., 20., -40., 0., 50., 0.], dtype=np.float32)
        self.high = np.array([35., 28., 28., 40., 1100., 180., 23.], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )

        self.state = None
        self.t = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _apply_action_setpoints(self, a_t):
        a_t = np.asarray(a_t, dtype=np.float32).reshape(-1)

        SAT_sp = float(
            np.clip(
                a_t[0],
                self.action_space.low[0],
                self.action_space.high[0],
            )
        )

        ZAT_sp = float(
            np.clip(
                a_t[1],
                self.action_space.low[1],
                self.action_space.high[1],
            )
        )

        return SAT_sp, ZAT_sp

    def _select_mode(self, T_zone, ZAT_sp):
        if T_zone < ZAT_sp - self.mode_deadband:
            return "heating"
        if T_zone > ZAT_sp + self.mode_deadband:
            return "cooling"
        return "neutral"

    def step(self, a_t):
        # Denormalize state
        s_t = self.state * (self.high - self.low) + self.low
        T_zone = float(s_t[1])

        # Advance time
        self.t += self.dt / 3600.0

        idx = min(int(self.t * 2), len(self.data) - 1)
        row = self.data.iloc[idx]

        T_out = float(row.Tout)
        Qsg = float(row.Qsg)
        Qint = float(row.Qint)
        Hour = float(row.Hour)
        T_cor = 24.0

        # Action -> setpoints
        SAT_sp, ZAT_sp = self._apply_action_setpoints(a_t)

        # Mode for logging only
        mode = self._select_mode(T_zone, ZAT_sp)

        # Reset PI integrator when ZAT changes
        if self.prev_ZAT_sp is None or ZAT_sp != self.prev_ZAT_sp:
            self.integral_error = 0.0

        self.prev_ZAT_sp = ZAT_sp

        x_room = s_t[:2].copy()

        total_energy = 0.0
        cool_energy_total = 0.0
        heat_energy_total = 0.0
        reheat_energy_total = 0.0
        fan_energy_total = 0.0

        damper_signal = float(self.damper_signal_prev)
        COPc = self.COP_rated

        reheat_signal_last = 0.0
        Q_reheat_last_W = 0.0

        u_base = np.array([T_cor, T_out, Qsg, Qint], dtype=np.float32)

        for _ in range(self.n_pi_loops):
            # Airflow from damper
            m_fan = self.m_dot_min + (damper_signal / 100.0) * (
                self.m_dot_max - self.m_dot_min
            )
            self.m_fan = m_fan

            # PI update
            error = T_zone - ZAT_sp
            raw = self.Kp * error + self.Ki * self.integral_error
            damper_signal = float(np.clip(raw, 0.0, 100.0))

            # Anti-windup
            if not ((raw >= 99.9 and error > 0) or (raw <= 0.1 and error < 0)):
                self.integral_error += error * self.pi_interval

            # Recompute airflow
            m_fan = self.m_dot_min + (damper_signal / 100.0) * (
                self.m_dot_max - self.m_dot_min
            )
            self.m_fan = m_fan

            # AHU heat transfer
            Q_air = self.capacity_scale * (
                m_fan * self.cp_air * (SAT_sp - T_zone)
            )

            # Reheat
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
                dtype=np.float32,
            )

            # Propagate room state
            x_room = self.A_pi @ x_room + self.B_pi @ u_model
            T_zone = float(x_room[1])

            # Energy calculation
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

            COPc = max(
                0.1,
                (self.COP_rated * capft * capfff) / (eirft * eirfff),
            )

            f_pl = (
                self.c_FAN[0]
                + self.c_FAN[1] * f_flow
                + self.c_FAN[2] * f_flow ** 2
                + self.c_FAN[3] * f_flow ** 3
            )

            Q_fan = f_pl * self.m_design * self.dP / (
                self.e_tot * self.rho_air
            )

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

        # Persist damper signal
        self.damper_signal_prev = float(damper_signal)

        # --------------------------------------------------
        # Modified comfort bound:
        # daytime strict, nighttime relaxed, both penalized
        # --------------------------------------------------
        T_now = float(x_room[1])

        if 7 <= Hour <= 20:
            lb = self.lb
            ub = self.ub
        else:
            lb = self.night_lb
            ub = self.night_ub

        Temp_exceed = 0.0

        if T_now < lb:
            Temp_exceed = lb - T_now
        elif T_now > ub:
            Temp_exceed = T_now - ub

        # Reward
        energy_norm = total_energy / max(self.E_ref, 1e-6)
        temp_norm = Temp_exceed / max(self.T_ref, 1e-6)

        reward = -(energy_norm + self.alpha * temp_norm)

        # Next normalized state
        s_ext = np.array([T_cor, T_out, Qsg, Qint, Hour], dtype=np.float32)

        self.state = (
            np.concatenate([x_room, s_ext]) - self.low
        ) / (self.high - self.low)

        done = self.t >= self.end

        info = {
            "Mode": mode,
            "SAT_sp": float(SAT_sp),
            "ZAT_sp_used": float(ZAT_sp),

            "m_fan": float(self.m_fan),
            "DamperSignal": float(damper_signal),

            "ReheatSignal": float(reheat_signal_last),
            "ReheatPct": float(100.0 * reheat_signal_last),
            "Q_reheat_W": float(Q_reheat_last_W),

            "TotalEnergy_kWh": float(total_energy),
            "CoolingEnergy_kWh": float(cool_energy_total),
            "HeatingEnergy_kWh": float(heat_energy_total),
            "ReheatEnergy_kWh": float(reheat_energy_total),
            "FanEnergy_kWh": float(fan_energy_total),

            "Hour": float(Hour),

            # New logging fields
            "lb": float(lb),
            "ub": float(ub),
            "ComfortLowerBound": float(lb),
            "ComfortUpperBound": float(ub),

            "TempExceed_degC": float(Temp_exceed),
            "Reward": float(reward),
            "EnergyNorm": float(energy_norm),
            "TempNorm": float(temp_norm),
        }

        return np.array(self.state, dtype=np.float32), float(reward), bool(done), info

    def reset(self):
        self.t = self.start
        self.integral_error = 0.0
        self.prev_ZAT_sp = None
        self.m_fan = self.m_dot_min
        self.damper_signal_prev = 0.0

        T_env_0 = 20.0
        T_zone_0 = 24.0
        T_cor = 24.0

        idx = min(int(self.start * 2), len(self.data) - 1)
        row = self.data.iloc[idx]

        T_out = float(row.Tout)
        Qsg = float(row.Qsg)
        Qint = float(row.Qint)
        Hour = float(row.Hour)

        self.state = (
            np.array(
                [T_env_0, T_zone_0, T_cor, T_out, Qsg, Qint, Hour],
                dtype=np.float32,
            ) - self.low
        ) / (self.high - self.low)

        return np.array(self.state, dtype=np.float32)