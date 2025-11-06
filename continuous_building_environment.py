__all__ = ["ContinuousBuildingControlEnvironment"]

import numpy as np
from scipy import signal
import pandas as pd
import gym
from gym import spaces, logger
from gym.utils import seeding

DATA_PATH = './data/'

# =======================================================================
# Environment class
# =======================================================================
class ContinuousBuildingControlEnvironment(gym.Env):
    """
    Building environment with inner PI control and optional RL supervisory layer.
    For now:
        - SAT is fixed to 26°C.
        - ZAT setpoint changes by time:
            26°C for occupied hours (7 <= Hour <= 20)
            18°C for unoccupied hours (else)
        - Includes anti-windup and integral reset when setpoint changes.
    """

    def __init__(self, data_file, dt=1800, start=0., end=10000.,
                 C_env=None, C_air=None, R_rc=None, R_oe=None, R_er=None,
                 lb_set=22., ub_set=24.):
        # --- Basic parameters ---
        self.dt = dt
        self.data = pd.read_csv(DATA_PATH + data_file)
        self.start = start
        self.end = end
        self.num_step = int((self.end - self.start) / (self.dt / 3600) + 1)

        # --- Thermal parameters ---
        self.C_env = C_env
        self.C_air = C_air
        self.R_rc = R_rc
        self.R_oe = R_oe
        self.R_er = R_er
        self.a_sol_env = 0.90303


        # --- State-space matrices (continuous) ---
        A = np.zeros((2, 2))
        B = np.zeros((2, 5))
        A[0, 0] = (-1. / self.C_env) * (1. / self.R_er + 1. / self.R_oe)
        A[0, 1] = 1. / (self.C_env * self.R_er)
        A[1, 0] = 1. / (self.C_air * self.R_er)
        A[1, 1] = (-1. / self.C_air) * (1. / self.R_er + 1. / self.R_rc)
        B[0, 1] = 1. / (self.C_env * self.R_oe)
        B[0, 2] = self.a_sol_env / self.C_env
        B[1, 0] = 1. / (self.C_air * self.R_rc)
        B[1, 2] = (1. - self.a_sol_env) / self.C_air
        B[1, 3] = 1. / self.C_air
        B[1, 4] = 1. / self.C_air
        
        # Store: Continuous Version
        self.Ac, self.Bc = A, B
                     
        # Discretize once for the outer timestep (dt)
        disc_dt = signal.StateSpace(self.Ac, self.Bc, np.array([[1, 0]]), np.zeros(5)).to_discrete(dt=self.dt)
        self.A, self.B = disc_dt.A, disc_dt.B

        # --- Comfort bounds ---
        self.beta = -0.05
        self.T_ref = 23.
        self.lb = lb_set
        self.ub = ub_set

        # --- HVAC parameters ---
        self.Tlv_cooling = 7.
        self.Tlv_heating = 35.
        self.E_cf_cooling = np.array([14.8187, -0.2538, 0.1814,
                                      -0.0003, -0.0021, 0.002])
        self.E_cf_heating = np.array([7.8885, 0.1809, -0.1568,
                                      0.001068, 0.0009938, -0.002674])
        self.cp_air = 1004
        self.m_dot_min = 0.080939
        self.m_design = 0.9264 * 0.4
        self.dP = 500
        self.e_tot = 0.6045
        self.rho_air = 1.225
        self.c_FAN = np.array([0.04076, 0.08804, -0.07293, 0.94374, 0])
        self.Qh_max = 1500
        self.m_dot_max = self.m_dot_min * 550 / 140
        self.seed()

        # --- Action/observation spaces ---
        self.action_space = spaces.Box(
            low=np.array([10.0, 15.0]),
            high=np.array([40.0, 30.0]),
            dtype=np.float32
        )
        self.low = np.array([10., 15., 21., -40., 0., 50., 0.])
        self.high = np.array([35., 28., 23., 40., 1100., 180., 23.])
        self.observation_space = spaces.Box(
            low=np.zeros(7), high=np.ones(7), dtype=np.float32
        )

        # --- PI control & supervisory settings ---
        self.Kp = 15
        self.Ki = 0.01
        self.integral_error = 0.0
        self.prev_ZAT_sp = None
        self.pi_interval = 60  # seconds
        self.m_fan = self.m_dot_min

        self.control_mode = "fixed"

        self.state = None
        self.t = None

    # ------------------------------------------------------------------
    # Utility control functions
    # ------------------------------------------------------------------
    def set_control_mode(self, mode: str):
        assert mode in ("fixed", "rl")
        self.control_mode = mode

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ------------------------------------------------------------------
    # Step function
    # ------------------------------------------------------------------
    def step(self, a_t):
        s_t = self.state * (self.high - self.low) + self.low
        T_zone = s_t[1]

        # --- Time and exogenous variables ---
        self.t += 0.5
        T_cor = 16.
        row = self.data.iloc[int(self.t * 2)]
        T_out, Qsg, Qint, Hour = row.Tout, row.Qsg, row.Qint, row.Hour

        # --- FIXED Time-dependent setpoints ---
        SAT_sp = 26.0
        ZAT_sp = 26.0 if 7 <= Hour <= 20 else 18.0

        # --- Reset integral if ZAT setpoint changed ---
        if self.prev_ZAT_sp is None or ZAT_sp != self.prev_ZAT_sp:
            self.integral_error = 0.0
        self.prev_ZAT_sp = ZAT_sp

        # --- Coefficients of performance ---
        EFh, EFc = self.E_cf_heating, self.E_cf_cooling
        Tlvh, Tlvc = self.Tlv_heating, self.Tlv_cooling
        COPh = 0.9
        COPc = EFc[0] + T_out * EFc[1] + Tlvh * EFc[2] + \
               (T_out ** 2) * EFc[3] + (Tlvc ** 2) * EFc[4] + T_out * Tlvc * EFc[5]

        # --- Inner PI loop with anti-windup ---
        n_loops = int(self.dt // self.pi_interval)
        if n_loops < 1:
            n_loops = 1
        
        # Re-discretize the continuous model for the PI interval
        disc_pi = signal.StateSpace(self.Ac, self.Bc, np.array([[1, 0]]), np.zeros(5)).to_discrete(dt=self.pi_interval)
        A_pi, B_pi = disc_pi.A, disc_pi.B
        
        # Initialize current state
        x_room = s_t[:2].copy()   # [T_env, T_zone]
        T_zone = float(x_room[1])
        total_energy = 0.0
        
        for _ in range(n_loops):
            # --- PI control signal ---
            error = ZAT_sp - T_zone
            u_int = self.Kp * error + self.Ki * self.integral_error
        
            # Anti-windup: stop integrating when saturated in same direction
            sat_hi = (u_int >= 99.9) and (error > 0)
            sat_lo = (u_int <= 0.1) and (error < 0)
            if not (sat_hi or sat_lo):
                self.integral_error = np.clip(self.integral_error + error * self.pi_interval, -1000, 1000)

            damper_raw = self.Kp * error + self.Ki * self.integral_error
            damper_signal = np.clip(damper_raw, 0.0, 100.0)
        
            # --- Airflow ---
            m_fan = self.m_dot_min + (damper_signal / 100.0) * (self.m_dot_max - self.m_dot_min)
            self.m_fan = m_fan
        
            # --- Zone heating power (W) ---
            u_t = m_fan * self.cp_air * (SAT_sp - T_zone)
        
            # --- Advance 3R2C model by one PI interval ---
            x_room = A_pi @ x_room + B_pi @ np.append(s_t[2:6], u_t)
            T_zone = float(x_room[1])  # updated zone temperature
        
            # --- Fan power (W) ---
            f_flow = m_fan / self.m_design
            f_pl = (self.c_FAN[0] + self.c_FAN[1]*f_flow + self.c_FAN[2]*f_flow**2
                    + self.c_FAN[3]*f_flow**3 + self.c_FAN[4]*f_flow**4)
            Q_fan = f_pl * self.m_design * self.dP / (self.e_tot * self.rho_air)
        
            # --- Electrical heating power (W) ---
            P_heat_elec = max(u_t, 0.0) / COPh
        
            # --- Accumulate energy (kWh) ---
            total_energy += ((P_heat_elec + Q_fan) / 1000.0) * (self.pi_interval / 3600.0)
        
        # --- Output updated state ---
        s_next_room = x_room
        
        # --- External states ---
        s_ext = np.array([T_cor, T_out, Qsg, Qint, Hour])

        # --- Comfort utility ---
        if 7 <= Hour <= 20:
            lb, ub = self.lb, self.ub
            if s_next_room[-1] < lb:
                Utility = -5.
                Temp_exceed = (lb - s_next_room[-1]) * 0.5
            elif s_next_room[-1] > ub:
                Utility = -5.
                Temp_exceed = (s_next_room[-1] - ub) * 0.5
            else:
                Utility, Temp_exceed = 0., 0.
        else:
            lb, ub, Utility, Temp_exceed = 15., 28., 0., 0.

        # --- Reward ---
        r = -1 * total_energy + Utility

        # --- Update state ---
        self.state = (np.concatenate([s_next_room, s_ext]) - self.low) / (self.high - self.low)
        done = bool((self.t + 0.5) > self.end)
        if done:
            logger.warn("You called step() after done=True; call reset() first.")

        return np.array(self.state), r, done, {
            'SAT_sp': SAT_sp,
            'ZAT_sp': ZAT_sp,
            'm_fan': self.m_fan,
            'Energy': total_energy,
            'Penalty': Utility / (-10),
            'Exceedance': Temp_exceed,
            'lb': lb,
            'ub': ub,
            'Hour': Hour,
            'DamperRaw': float(damper_raw),
            'DamperSignal': float(damper_signal)
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self.t = self.start
        self.integral_error = 0.0
        self.m_fan = self.m_dot_min
        self.prev_ZAT_sp = None
        T_env_0, T_air_0, T_cor = 12., 12., 12.
        row = self.data.iloc[int(self.t * 2)]
        T_out, Qsg, Qint, Hour = row.Tout, row.Qsg, row.Qint, row.Hour
        self.state = (np.array([T_env_0, T_air_0, T_cor, T_out, Qsg, Qint, Hour])
                      - self.low) / (self.high - self.low)
        return np.array(self.state)

