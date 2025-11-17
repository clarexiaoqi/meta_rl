import matplotlib.pyplot as plt
import numpy as np
from continuous_building_environment import ContinuousBuildingControlEnvironment

# Initialize environment
env = ContinuousBuildingControlEnvironment(
    data_file='weather_data_2013_to_2017_summer_pandas.csv',
    C_env=1.5e6,
    C_air=1.0e6,
    R_rc=0.06,
    R_oe=1.2,
    R_er=0.4
)

# Simulation settings
n_steps = 96
dt_hours = env.dt / 3600
time = np.arange(n_steps) * dt_hours

# Storage
T_env_hist, T_zone_hist, T_cor_hist, T_out_hist = [], [], [], []
ZAT_sp_hist, SAT_sp_hist = [], []
m_fan_hist, damper_hist = [], []
hour_hist, cop_hist = [], []
Qsg_hist = []

total_E, cool_E, heat_E, fan_E = [], [], [], []

state = env.reset()

# Simulation loop
for step in range(n_steps):
    next_state, reward, done, info = env.step([0, 0])

    s_real = next_state * (env.high - env.low) + env.low
    T_env, T_zone, T_cor, T_out = s_real[:4]
    Qsg = s_real[4]

    T_env_hist.append(T_env)
    T_zone_hist.append(T_zone)
    T_cor_hist.append(T_cor)
    T_out_hist.append(T_out)
    Qsg_hist.append(Qsg)

    ZAT_sp_hist.append(info["ZAT_sp"])
    SAT_sp_hist.append(info["SAT_sp"])

    damper_hist.append(info["DamperSignal"])
    m_fan_hist.append(info["m_fan"])
    cop_hist.append(info["COPc"])
    hour_hist.append(info["Hour"])

    total_E.append(info["TotalEnergy"])
    cool_E.append(info["CoolingEnergy"])
    heat_E.append(info["HeatingEnergy"])
    fan_E.append(info["FanEnergy"])

    if done:
        break

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# 1) Temperatures + Qsg
ax1 = axs[0]
ax1.plot(time, T_zone_hist, label='Zone Temp')
ax1.plot(time, T_env_hist, label='Envelope Temp')
ax1.plot(time, T_cor_hist, label='Corridor Temp')
ax1.plot(time, T_out_hist, label='Outdoor Temp')
ax1.plot(time, ZAT_sp_hist, '--', label='ZAT Setpoint')
ax1.plot(time, SAT_sp_hist, ':', label='SAT Setpoint')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('PI Cooling Control with Energy Plots')
ax1.legend(loc='upper left')

ax1b = ax1.twinx()
ax1b.plot(time, Qsg_hist, 'k--', alpha=0.6, label='Qsg')
ax1b.set_ylabel('Qsg (W)')
ax1b.legend(loc='upper right')

# 2) Damper + Fan flow
ax2 = axs[1]
ax2b = ax2.twinx()
ax2.plot(time, damper_hist, label='Damper (%)')
ax2b.plot(time, m_fan_hist, linestyle=':', label='Fan Flow (kg/s)')
ax2.set_ylabel('Damper (%)')
ax2b.set_ylabel('Fan (kg/s)')
ax2.legend(loc='upper left')
ax2b.legend(loc='upper right')

# 3) Energy
ax3 = axs[2]
ax3.plot(time, total_E, label='Total Energy (kWh)')
ax3.plot(time, cool_E, label='Cooling (kWh)')
ax3.plot(time, heat_E, label='Heating (kWh)')
ax3.plot(time, fan_E, label='Fan (kWh)')
ax3.set_ylabel('Energy (kWh)')
ax3.set_xlabel('Time (h)')
ax3.legend()

plt.tight_layout()
plt.show()
