__all__ = ["ContinuousBuildingControlEnvironment"]

import numpy as np
from scipy import signal
import pandas as pd
import gym
from gym import spaces, logger
from gym.utils import seeding

DATA_PATH = "./data/"


class ContinuousBuildingControlEnvironment(gym.Env):
    """
    HVAC environment with:
      - 3R2C thermal model
      - Inner PI loop (pi_interval, default 300 s)
      - Outer supervisory timestep dt (default 1800 s = 30 min)
      - Air-based cooling AND air-based heating (mode-dependent SAT/ZAT)
      - Terminal reheat enabled ONLY in cooling mode
      - York DNZ060 cooling performance curves for cooling COP
      - Comfort penalty (Utility)
    """

    def __init__(
        self,
        data_file,
        dt=1800.0,
        start=0.0,
        end=10000.0,
        C_env=None,
        C_air=None,
        R_rc=None,
        R_oe=None,
        R_er=None,
        lb_set=22.0,
        ub_set=24.0,
        # -------- fixed setpoints (for now) --------
        SAT_cool=16.0,
        ZAT_cool=20.0,
        SAT_heat=35.0,
        ZAT_heat=22.0,
        mode_deadband=0,
    ):
        # =============================
        # Basic configuration
        # =============================
        self.dt = float(dt)
        self.data = pd.read_csv(DATA_PATH + data_file)
        self.start = float(start)
        self.end = float(end)

        # Thermal model parameters
        if C_env is None or C_air is None or R_rc is None or R_oe is None or R_er is None:
            raise ValueError("C_env, C_air, R_rc, R_oe, R_er must be provided (not None).")
        self.C_env = float(C_env)
        self.C_air = float(C_air)
        self.R_rc = float(R_rc)
        self.R_oe = float(R_oe)
        self.R_er = float(R_er)
        self.a_sol_env = 0.90303

        # =============================
        # PI Controller settings
        # =============================
        self.Kp = 15.0
        self.Ki = 0.02
        self.integral_error = 0.0
        self.prev_ZAT_sp = None
        self.pi_interval = 60.0 * 5.0  # 300 s
        self.m_fan = None

        # =============================
        # Fixed setpoints (cooling/heating)
        # =============================
        self.SAT_cool = float(SAT_cool)
        self.ZAT_cool = float(ZAT_cool)
        self.SAT_heat = float(SAT_heat)
        self.ZAT_heat = float(ZAT_heat)
        self.mode_deadband = float(mode_deadband)

        # =============================
        # Reheat constants
        # =============================
        self.Qh_reheat_max = 1500.0  # W (terminal reheat)
        self.eta_reheat = 0.9        # for reheat + (simple) air-based heating energy accounting

        # =============================
        # 3R2C continuous model
        # =============================
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

        # Outer discrete model
        disc_dt = signal.StateSpace(
            self.Ac, self.Bc,
            np.array([[1.0, 0.0]]),
            np.zeros(5)
        ).to_discrete(dt=self.dt)
        self.A_dt, self.B_dt = disc_dt.A, disc_dt.B

        # Inner PI discrete model
        disc_pi = signal.StateSpace(
            self.Ac, self.Bc,
            np.array([[1.0, 0.0]]),
            np.zeros(5)
        ).to_discrete(dt=self.pi_interval)
        self.A_pi, self.B_pi = disc_pi.A, disc_pi.B

        # timestep matching
        self.n_pi_loops = int(self.dt // self.pi_interval)
        if self.n_pi_loops < 1:
            self.n_pi_loops = 1
        self.dt_hr_pi = self.pi_interval / 3600.0

        # Comfort bounds (occupied hours only)
        self.lb = float(lb_set)
        self.ub = float(ub_set)

        # ============================================================
        # York Affinity DNZ060 Cooling Curves (for cooling COP only)
        # ============================================================
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

        # =============================
        # HVAC constants
        # =============================
        self.cp_air = 1004.0
        self.m_dot_min = 0.080939
        self.m_design = 0.9264 * 0.4
        self.m_dot_max = self.m_dot_min * 550.0 / 140.0

        self.dP = 500.0
        self.e_tot = 0.6045
        self.rho_air = 1.225
        self.c_FAN = np.array([0.04076, 0.08804, -0.07293, 0.94374, 0.0])

        # scale to represent multi-zone aggregate (your existing choice)
        self.capacity_scale = 1.0 / 3.0

        # =============================
        # Action/State Spaces
        # =============================
        # Keeping the original Box, but in this fixed-setpoint version we ignore a_t.
        self.action_space = spaces.Box(
            low=np.array([10.0, 15.0]),
            high=np.array([40.0, 30.0]),
            dtype=np.float32,
        )

        # State = [T_env, T_zone, T_cor, T_out, Qsg, Qint, Hour]
        self.low = np.array([10., 15., 20., -40., 0., 50., 0.])
        self.high = np.array([35., 28., 28., 40., 1100., 180., 23.])

        self.observation_space = spaces.Box(
            low=np.zeros(7),
            high=np.ones(7),
            dtype=np.float32,
        )

        self.state = None
        self.t = None

        self.seed()

    # --------------------------------------------------------------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # --------------------------------------------------------------
    def _select_mode_and_setpoints(self, T_zone):
        """
        Supervisory mode selection (discrete):
          - heating if below heating setpoint - deadband
          - cooling if above cooling setpoint + deadband
          - neutral otherwise (default to cooling setpoints)
        """
        if T_zone < self.ZAT_heat - self.mode_deadband:
            return "heating", self.SAT_heat, self.ZAT_heat
        if T_zone > self.ZAT_cool + self.mode_deadband:
            return "cooling", self.SAT_cool, self.ZAT_cool
        return "neutral", self.SAT_cool, self.ZAT_cool

    # ============================================================
    def step(self, a_t):
        # -----------------------------
        # Denormalize state
        # -----------------------------
        s_t = self.state * (self.high - self.low) + self.low
        T_zone = float(s_t[1])

        # Outer time advance (hours)
        self.t += self.dt / 3600.0

        # Exogenous inputs
        row = self.data.iloc[int(self.t * 2)]
        T_out, Qsg, Qint, Hour = row.Tout, row.Qsg, row.Qint, row.Hour
        T_cor = 24.0

        # Supervisory mode selection (once per outer step)
        mode, SAT_sp, ZAT_sp = self._select_mode_and_setpoints(T_zone)

        # Reset PI integrator on setpoint change
        if self.prev_ZAT_sp is None or ZAT_sp != self.prev_ZAT_sp:
            self.integral_error = 0.0
        self.prev_ZAT_sp = ZAT_sp

        x_room = s_t[:2].copy()
        n_loops = self.n_pi_loops

        total_energy = 0.0
        cool_energy_total = 0.0
        heat_energy_total = 0.0
        reheat_energy_total = 0.0
        fan_energy_total = 0.0

        damper_signal = 0.0
        COPc = self.COP_rated

        u_base = np.array([T_cor, T_out, Qsg, Qint])

        # ============================================================
        # Inner PI loop
        # ============================================================
        for _ in range(n_loops):
            # ---------------------------------
            # Airflow from PI damper
            # ---------------------------------
            m_fan = self.m_dot_min + (damper_signal / 100.0) * (self.m_dot_max - self.m_dot_min)
            self.m_fan = m_fan

            # --- Effective airflow percentage (for plotting only) ---
            self.damper_eff = 100.0 * (m_fan / self.m_design)

            # ---------------------------------
            # PI control (zone temperature -> damper)
            # ---------------------------------
            error = T_zone - ZAT_sp
            raw = self.Kp * error + self.Ki * self.integral_error
            damper_signal = float(np.clip(raw, 0.0, 100.0))

            # anti-windup
            if not ((raw >= 99.9 and error > 0) or (raw <= 0.1 and error < 0)):
                self.integral_error += error * self.pi_interval

            # ---------------------------------
            # Air-based HVAC heat transfer (works for both cooling/heating)
            # Q_air < 0 => cooling, Q_air > 0 => heating
            # ---------------------------------
            Q_air = self.capacity_scale * (m_fan * self.cp_air * (SAT_sp - T_zone))

            # --------------------------------------------------
            # Terminal reheat ONLY at minimum airflow
            # --------------------------------------------------
            at_min_flow = damper_signal <= 1.0

            if (
                    mode in ["cooling", "neutral"]
                    and T_zone < ZAT_sp
                    and at_min_flow
            ):
                reheat_signal = np.clip((ZAT_sp - T_zone), 0.0, 1.0)
                Q_reheat = reheat_signal * self.Qh_reheat_max
            else:
                Q_reheat = 0.0

            # ---------------------------------
            # Single thermal update
            # ---------------------------------
            u_total = Q_air + Q_reheat
            u_model = np.array([u_base[0], u_base[1], u_base[2], u_base[3], u_total])

            x_room = self.A_pi @ x_room + self.B_pi @ u_model
            T_zone = float(x_room[1])

            # ========================================================
            # ENERGY CALCULATIONS
            # ========================================================
            f_flow = max(0.05, m_fan / self.m_design)

            capfff = self.capfff["C1"] + self.capfff["C2"] * f_flow + self.capfff["C3"] * f_flow ** 2
            eirfff = self.eirfff["C1"] + self.eirfff["C2"] * f_flow + self.eirfff["C3"] * f_flow ** 2

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

            # Cooling compressor power (only when Q_air is cooling)
            P_cool = max(-Q_air, 0.0) / COPc

            # Heating via hot supply air (simple efficiency model)
            P_heat = max(Q_air, 0.0) / self.eta_reheat

            # Terminal reheat power
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

        # ============================================================
        # Comfort penalty
        # ============================================================
        T_now = x_room[1]
        Utility = 0.0
        Temp_exceed = 0.0

        if 7 <= Hour <= 20:
            if T_now < self.lb:
                Utility = -5.0
                Temp_exceed = self.lb - T_now
            elif T_now > self.ub:
                Utility = -5.0
                Temp_exceed = T_now - self.ub

        reward = -total_energy + Utility

        # next state
        s_ext = np.array([T_cor, T_out, Qsg, Qint, Hour])
        self.state = (np.concatenate([x_room, s_ext]) - self.low) / (self.high - self.low)

        done = self.t >= self.end

        info = {
            "Mode": mode,
            "SAT_sp": SAT_sp,
            "ZAT_sp": ZAT_sp,
            "m_fan": self.m_fan,
            "COPc": COPc,
            "DamperSignal": damper_signal,
            "DamperEff": self.damper_eff,
            "TotalEnergy": total_energy,
            "CoolingEnergy": cool_energy_total,
            "HeatingEnergy": heat_energy_total,
            "ReheatEnergy": reheat_energy_total,
            "FanEnergy": fan_energy_total,
            "Hour": Hour,
            "Utility": Utility,
            "TempExceed": Temp_exceed,
        }

        return np.array(self.state), reward, done, info

    # ============================================================
    def reset(self):
        self.t = self.start
        self.integral_error = 0.0
        self.prev_ZAT_sp = None
        self.m_fan = self.m_dot_min

        T_env_0 = 20.0
        T_zone_0 = 24.0
        T_cor = 24.0

        row = self.data.iloc[int(self.start * 2)]
        T_out, Qsg, Qint, Hour = row.Tout, row.Qsg, row.Qint, row.Hour

        self.state = (
            np.array([T_env_0, T_zone_0, T_cor, T_out, Qsg, Qint, Hour]) - self.low
        ) / (self.high - self.low)

        return np.array(self.state)
