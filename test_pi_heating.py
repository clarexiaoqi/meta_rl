import matplotlib.pyplot as plt
import numpy as np
from continuous_building_environment import ContinuousBuildingControlEnvironment

# ============================================================
# 1. Initialize environment
# ============================================================
env = ContinuousBuildingControlEnvironment(
    data_file='weather_data_2013_to_2017_winter_pandas.csv',  # must exist in ./data/
    C_env=1.5e6,
    C_air=1.0e6,
    R_rc=0.06, #0.15
    R_oe=1.2, #1.2
    R_er=0.4 #0.4
)

env.set_control_mode("fixed")  # PI-only test

# ============================================================
# 2. Simulation parameters
# ============================================================
n_steps = 96        # 2 days (dt = 1800 s → 48 steps/day)
dt_hours = env.dt / 3600
time = np.arange(n_steps) * dt_hours

# Storage arrays
T_env_hist, T_zone_hist, T_cor_hist, T_out_hist = [], [], [], []
ZAT_sp_hist, SAT_sp_hist = [], []
m_fan_hist, damper_hist = [], []
hour_hist = []

# ============================================================
# 3. Run simulation
# ============================================================
state = env.reset()
for step in range(n_steps):
    next_state, reward, done, info = env.step([0, 0])  # action ignored in fixed mode

    # Extract real (unnormalized) state
    s_real = next_state * (env.high - env.low) + env.low
    T_env, T_zone, T_cor, T_out = s_real[0], s_real[1], s_real[2], s_real[3]
    Hour = info['Hour']

    # PI variables
    error = info['ZAT_sp'] - T_zone
    damper_signal = np.clip(env.Kp * error + env.Ki * env.integral_error, 0, 100)

    # Record data
    T_env_hist.append(T_env)
    T_zone_hist.append(T_zone)
    T_cor_hist.append(T_cor)
    T_out_hist.append(T_out)
    ZAT_sp_hist.append(info['ZAT_sp'])
    SAT_sp_hist.append(info['SAT_sp'])
    m_fan_hist.append(info['m_fan'])
    damper_hist.append(damper_signal)
    hour_hist.append(Hour)

    print(f"Step {step:03d} | Hour={Hour:4.1f} | "
          f"T_env={T_env:.2f}°C | T_zone={T_zone:.2f}°C | "
          f"T_cor={T_cor:.2f}°C | T_out={T_out:.1f}°C | "
          f"ZAT_sp={info['ZAT_sp']:.1f}°C | SAT_sp={info['SAT_sp']:.1f}°C | "
          f"m_fan={info['m_fan']:.4f} kg/s | Damper={damper_signal:.1f}%")

    if done:
        break

# ============================================================
# 4. Plot results (2 plots total)
# ============================================================
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# ------------------------------------------------------------
# (1) Temperatures
# ------------------------------------------------------------
ax1 = axs[0]

ax1.plot(time, T_zone_hist, label='Zone Temp', color='black')
ax1.plot(time, T_env_hist, label='Envelope Temp', color='tab:cyan')
ax1.plot(time, T_cor_hist, label='Corridor Temp', color='tab:pink')
ax1.plot(time, T_out_hist, label='Outdoor Temp', color='gray', alpha=0.6)
ax1.plot(time, ZAT_sp_hist, 'r--', label='ZAT Setpoint (°C)')
ax1.plot(time, SAT_sp_hist, 'orange', linestyle=':', label='SAT (°C)')

ax1.set_ylabel('Temperature (°C)')
ax1.set_title('PI Control Test: Heating Mode with Time-Varying ZAT')
ax1.legend(loc='upper right')

# ------------------------------------------------------------
# (2) Damper + Fan mass flow (dual y-axis)
# ------------------------------------------------------------
ax3 = axs[1]
ax4 = ax3.twinx()

ax3.plot(time, damper_hist, color='tab:orange', label='Damper Signal (%)')
ax4.plot(time, m_fan_hist, color='tab:green', linestyle=':', label='Fan Airflow (kg/s)')

ax3.set_ylabel('Damper Signal (%)', color='tab:orange')
ax4.set_ylabel('Fan Airflow (kg/s)', color='tab:green')
ax3.set_xlabel('Time (hours)')

# Combine legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax4.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.show()
