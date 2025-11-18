__all__ = ["ContinuousBuildingControlEnvironment"]

import numpy as np
from scipy import signal
import pandas as pd
import gym
from gym import spaces, logger
from gym.utils import seeding

DATA_PATH = './data/'


class ContinuousBuildingControlEnvironment(gym.Env):
    """
    Cooling-only HVAC environment with:
      - 3R2C thermal model
      - PI loop (60*5-second internal timestep)
      - Dynamic SAT reset (Option C)
      - York DNZ060 cooling performance curves (CAPFT, EIRFT, CAPFFF, EIRFFF)
    """

    def __init__(
        self,
        data_file,
        dt=1800,
        start=0.0,
        end=10000.0,
        C_env=None,
        C_air=None,
        R_rc=None,
        R_oe=None,
        R_er=None,
        lb_set=22.0,
        ub_set=24.0,
    ):
        # =============================
        # Basic configuration
        # =============================
        self.dt = dt
        self.data = pd.read_csv(DATA_PATH + data_file)
        self.start = start
        self.end = end

        # Thermal model parameters
        self.C_env = C_env
        self.C_air = C_air
        self.R_rc = R_rc
        self.R_oe = R_oe
        self.R_er = R_er
        self.a_sol_env = 0.90303

        # =============================
        # PI Controller settings
        # =============================
        self.Kp = 10.0
        self.Ki = 0.01
        self.integral_error = 0.0
        self.prev_ZAT_sp = None
        self.pi_interval = 60.0*5
        self.m_fan = None

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

        # Outer (1800s) discrete model
        disc_dt = signal.StateSpace(
            self.Ac, self.Bc,
            np.array([[1.0, 0.0]]),
            np.zeros(5)
        ).to_discrete(dt=self.dt)
        self.A_dt, self.B_dt = disc_dt.A, disc_dt.B

        # Inner (60*5s) PI discrete model
        disc_pi = signal.StateSpace(
            self.Ac, self.Bc,
            np.array([[1.0, 0.0]]),
            np.zeros(5)
        ).to_discrete(dt=self.pi_interval)
        self.A_pi, self.B_pi = disc_pi.A, disc_pi.B

        # Comfort bounds
        self.lb = lb_set
        self.ub = ub_set

        # ============================================================
        # York Affinity DNZ060 Cooling Curves (UPDATED)
        # ============================================================

        # CAPFT (Capacity Modifier vs Temperature) – biquadratic
        # x = entering wet-bulb temperature (°C)
        # y = outdoor dry-bulb temperature (°C)
        self.capft = {
            "C1": 1.2343140,
            "C2": -0.0398816,
            "C3": 0.0019354,
            "C4": 0.0062114,
            "C5": -0.0001247,
            "C6": -0.0003619
        }

        # EIRFT (EIR Modifier vs Temperature) – biquadratic
        self.eirft = {
            "C1": -0.1272387,
            "C2": 0.0848124,
            "C3": -0.0021062,
            "C4": -0.0085792,
            "C5": 0.0007783,
            "C6": -0.0005585
        }

        # CAPFFF (Capacity Modifier vs Flow Fraction) – quadratic
        # f_flow = m_fan / m_design
        self.capfff = {
            "C1": 1.2527302,
            "C2": -0.7182445,
            "C3": 0.4623738
        }
        self.capfff_min = 0.714
        self.capfff_max = 1.2

        # EIRFFF (EIR Modifier vs Flow Fraction) – quadratic
        self.eirfff = {
            "C1": 0.6529892,
            "C2": 0.8193151,
            "C3": -0.4617716
        }
        self.eirfff_min = 0.714
        self.eirfff_max = 1.2

        # =============================
        # HVAC constants
        # =============================
        self.cp_air = 1004.0
        self.COP_rated = 4.24  # updated rated COP for York DNZ060

        self.m_dot_min = 0.080939
        self.m_design = 0.9264 * 0.4
        self.m_dot_max = self.m_dot_min * 550.0 / 140.0

        self.dP = 500.0
        self.e_tot = 0.6045
        self.rho_air = 1.225

        self.c_FAN = np.array([0.04076, 0.08804, -0.07293, 0.94374, 0.0])

        self.capacity_scale = 1.0 / 3.0 # Cooling capacity scaling (RTU serves 3 zones; model has 1 zone)

        # =============================
        # Action/Observation Spaces
        # =============================
        self.action_space = spaces.Box(
            low=np.array([10.0, 15.0]),
            high=np.array([40.0, 30.0]),
            dtype=np.float32,
        )

        # state = [T_env, T_zone, T_cor, T_out, Qsg, Qint, Hour]
        self.low = np.array([10., 15., 20., -40., 0., 50., 0.])
        self.high = np.array([35., 28., 28., 40., 1100., 180., 23.])

        self.observation_space = spaces.Box(
            low=np.zeros(7), high=np.ones(7), dtype=np.float32
        )

        self.control_mode = "fixed"
        self.state = None
        self.t = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ============================================================
    # STEP
    # ============================================================
    def step(self, a_t):

        s_t = self.state * (self.high - self.low) + self.low
        T_zone = float(s_t[1])

        # Time advance
        self.t += self.dt / 3600.0

        row = self.data.iloc[int(self.t * 2)]
        T_out, Qsg, Qint, Hour = row.Tout, row.Qsg, row.Qint, row.Hour
        T_cor = 24.0

        # =============================
        # ZAT Setpoint
        # =============================
        ZAT_sp = 24.0 if 7 <= Hour <= 20 else 28.0

        # Reset integrator on ZAT change
        if self.prev_ZAT_sp is None or ZAT_sp != self.prev_ZAT_sp:
            self.integral_error = 0.0
        self.prev_ZAT_sp = ZAT_sp

        # number of PI iterations
        n_loops = int(self.dt // self.pi_interval)
        if n_loops < 1:
            n_loops = 1

        x_room = s_t[:2].copy()

        total_energy = 0.0
        cool_energy_total = 0.0
        heat_energy_total = 0.0
        fan_energy_total = 0.0

        damper_signal = 0.0
        COPc = self.COP_rated

        u_base = np.array([T_cor, T_out, Qsg, Qint])

        # ============================================================
        # Internal PI loop (every 60*5 sec)
        # ============================================================
        for _ in range(n_loops):

            # 1) PI controller (cooling)
            error = T_zone - ZAT_sp
            raw = self.Kp * error + self.Ki * self.integral_error
            damper_signal = float(np.clip(raw, 0.0, 100.0))

            if not ((raw >= 99.9 and error > 0) or (raw <= 0.1 and error < 0)):
                self.integral_error += error * self.pi_interval

            SAT_sp = 18.0

            # 2) Airflow
            m_fan = self.m_dot_min + (damper_signal / 100.0) * (self.m_dot_max - self.m_dot_min)
            self.m_fan = m_fan

            # 3) HVAC thermal load
            u_t = self.capacity_scale * (m_fan * self.cp_air * (SAT_sp - T_zone))

            # 4) Update 3R2C model
            inputs = np.append(u_base, u_t)
            x_room = self.A_pi @ x_room + self.B_pi @ inputs
            T_zone = float(x_room[1])

            # ---------------------------------------------------------
            # 5) York DNZ060 Cooling Curve Evaluation
            # ---------------------------------------------------------
            # Flow fraction
            f_flow = m_fan / self.m_design
            f_flow = np.clip(f_flow, self.capfff_min, self.capfff_max)

            # CAPFFF (quadratic)
            capfff = (
                    self.capfff["C1"]
                    + self.capfff["C2"] * f_flow
                    + self.capfff["C3"] * f_flow ** 2
            )

            # EIRFFF (quadratic)
            eirfff = (
                    self.eirfff["C1"]
                    + self.eirfff["C2"] * f_flow
                    + self.eirfff["C3"] * f_flow ** 2
            )

            # CAPFT
            capft = (
                self.capft["C1"]
                + self.capft["C2"] * T_out
                + self.capft["C3"] * T_out**2
                + self.capft["C4"] * SAT_sp
                + self.capft["C5"] * SAT_sp**2
                + self.capft["C6"] * T_out * SAT_sp
            )

            # EIRFT
            eirft = (
                self.eirft["C1"]
                + self.eirft["C2"] * T_out
                + self.eirft["C3"] * T_out**2
                + self.eirft["C4"] * SAT_sp
                + self.eirft["C5"] * SAT_sp**2
                + self.eirft["C6"] * T_out * SAT_sp
            )

            # COP
            COPc = (self.COP_rated * capft * capfff) / (eirft * eirfff)
            COPc = max(0.1, COPc)

            # ---------------------------------------------------------
            # 6) Energy calculations
            # ---------------------------------------------------------
            f_flow = m_fan / self.m_design
            f_pl = (
                self.c_FAN[0]
                + self.c_FAN[1] * f_flow
                + self.c_FAN[2] * f_flow**2
                + self.c_FAN[3] * f_flow**3
            )

            Q_fan = f_pl * self.m_design * self.dP / (self.e_tot * self.rho_air)

            P_cool = max(-u_t, 0.0) / COPc
            P_heat = max(u_t, 0.0) / 0.9

            dt_hr = self.pi_interval / 3600.0

            cool_step = (P_cool / 1000.0) * dt_hr
            heat_step = (P_heat / 1000.0) * dt_hr
            fan_step = (Q_fan / 1000.0) * dt_hr

            total_energy += cool_step + heat_step + fan_step
            cool_energy_total += cool_step
            heat_energy_total += heat_step
            fan_energy_total += fan_step

        # =============================
        # store results
        # =============================
        s_ext = np.array([T_cor, T_out, Qsg, Qint, Hour])

        reward = -total_energy

        self.state = (
            np.concatenate([x_room, s_ext]) - self.low
        ) / (self.high - self.low)
        done = (self.t >= self.end)

        info = {
            "SAT_sp": SAT_sp,
            "ZAT_sp": ZAT_sp,
            "m_fan": self.m_fan,
            "COPc": COPc,
            "DamperSignal": damper_signal,
            "TotalEnergy": total_energy,
            "CoolingEnergy": cool_energy_total,
            "HeatingEnergy": heat_energy_total,
            "FanEnergy": fan_energy_total,
            "Hour": Hour,
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
            np.array([T_env_0, T_zone_0, T_cor, T_out, Qsg, Qint, Hour])
            - self.low
        ) / (self.high - self.low)

        return np.array(self.state)
