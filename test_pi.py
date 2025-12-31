import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from continuous_building_environment import ContinuousBuildingControlEnvironment

# ============================================================
# SELECT EXPERIMENT (1–5)
# ============================================================
EXPERIMENT_ID = 6   # <<< CHANGE HERE\

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================
OCC_START = 7
OCC_END   = 20
STEPS_PER_DAY = 48          # 30-min timestep
N_DAYS = 9
rng = np.random.default_rng(seed=42)

# ---- Define ZAT per day (occupied hours) ----
if EXPERIMENT_ID in [1, 2, 3]:
    # Fixed ZAT = 20°C for all days
    ZAT_day = np.full(N_DAYS, 20.0)

elif EXPERIMENT_ID in [4, 5, 6]:
    # Day 1–9: 18 → 26 °C
    ZAT_day = np.arange(18, 27)

else:
    raise ValueError("EXPERIMENT_ID must be between 1 and 6")

# ---- Define SAT for cooling (explicit per experiment) ----
if EXPERIMENT_ID in [1, 4]:
    SAT_COOL = 17.0
elif EXPERIMENT_ID in [2, 5]:
    SAT_COOL = 12.0
elif EXPERIMENT_ID in [3, 6]:
    SAT_COOL = 7.0
else:
    raise ValueError("EXPERIMENT_ID must be between 1 and 6")

# ---- Fixed heating values ----
SAT_HEAT = 35.0
ZAT_HEAT = -100 #No heating for test

print(f"Experiment {EXPERIMENT_ID}")
print(f"Cooling SAT = {SAT_COOL}")
print(f"Daily occupied ZATs = {ZAT_day}")

# ============================================================
# Initialize environment (UNCHANGED)
# ============================================================
env = ContinuousBuildingControlEnvironment(
    data_file='weather_data_2013_to_2017_summer_pandas.csv',
    C_env=3.1996e6,
    C_air=3.5187e5,
    R_rc=0.00706,
    R_oe=0.02707,
    R_er=0.00369
)

# ============================================================
# Simulation settings
# ============================================================
n_steps = STEPS_PER_DAY * N_DAYS
dt_hours = env.dt / 3600
time = np.arange(n_steps) * dt_hours

# ============================================================
# Storage for plots
# ============================================================
T_env_hist, T_zone_hist, T_cor_hist, T_out_hist = [], [], [], []
ZAT_sp_hist, SAT_sp_hist = [], []
m_fan_hist, damper_hist = [], []
hour_hist, cop_hist = [], []
Qsg_hist = []

total_E, cool_E, reheat_E, heat_E, fan_E = [], [], [], [], []

# Storage for CSV logging
log_rows = []

state = env.reset()

# ============================================================
# Simulation loop
# ============================================================
for step in range(n_steps):

    # --------------------------------------------------------
    # Day / hour logic (THIS IS THE KEY ADDITION)
    # --------------------------------------------------------
    day_idx = step // STEPS_PER_DAY
    hour = env.data.iloc[int(env.t * 2)].Hour

    # Default (non-occupied)
    env.ZAT_cool = 26.0
    env.SAT_cool = SAT_COOL

    # Occupied hours → experiment ZAT
    if OCC_START <= hour <= OCC_END:
        env.ZAT_cool = ZAT_day[day_idx]

    # Heating fixed
    env.ZAT_heat = ZAT_HEAT
    env.SAT_heat = SAT_HEAT

    # --------------------------------------------------------
    # Step environment
    # --------------------------------------------------------
    next_state, reward, done, info = env.step([0, 0])

    s_real = next_state * (env.high - env.low) + env.low
    T_env, T_zone, T_cor, T_out = s_real[:4]
    Qsg = s_real[4]

    # --------------------------------------------------------
    # LOG TO CSV
    # --------------------------------------------------------
    log_rows.append({
        "Day":            day_idx + 1,
        "Hour":           info["Hour"],
        "T_zone":         T_zone,
        "T_env":          T_env,
        "T_cor":          T_cor,
        "T_out":          T_out,
        "ZAT_sp":         info["ZAT_sp"],
        "SAT_sp":         info["SAT_sp"],
        "Qsg":            Qsg,
        "Damper":         info["DamperEff"],
        "FanFlow":        info["m_fan"],
        "TotalEnergy":    info["TotalEnergy"],
        "Cooling":        info["CoolingEnergy"],
        "Reheat":         info["ReheatEnergy"],
        "Heating":        info["HeatingEnergy"],
        "Fan":            info["FanEnergy"],
    })

    # --------------------------------------------------------
    # For plotting
    # --------------------------------------------------------
    T_env_hist.append(T_env)
    T_zone_hist.append(T_zone)
    T_cor_hist.append(T_cor)
    T_out_hist.append(T_out)
    Qsg_hist.append(Qsg)

    ZAT_sp_hist.append(info["ZAT_sp"])
    SAT_sp_hist.append(info["SAT_sp"])

    damper_hist.append(info["DamperEff"])
    m_fan_hist.append(info["m_fan"])
    cop_hist.append(info["COPc"])
    hour_hist.append(info["Hour"])

    total_E.append(info["TotalEnergy"])
    cool_E.append(info["CoolingEnergy"])
    reheat_E.append(info["ReheatEnergy"])
    heat_E.append(info["HeatingEnergy"])
    fan_E.append(info["FanEnergy"])

    if done:
        break

# ============================================================
# Save CSV
# ============================================================
df_log = pd.DataFrame(log_rows)
csv_name = f"pi_test_log_exp{EXPERIMENT_ID}.csv"
df_log.to_csv(csv_name, index=False)
print(f"Saved log to {csv_name}")

# ============================================================
# Plot
# ============================================================
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
ax1.set_title(f'PI Cooling Control – Experiment {EXPERIMENT_ID}')
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
ax3.plot(time, reheat_E, label='Reheat (kWh)')
ax3.plot(time, heat_E, label='Heating (kWh)')
ax3.plot(time, fan_E, label='Fan (kWh)')
ax3.set_ylabel('Energy (kWh)')
ax3.set_xlabel('Time (h)')
ax3.legend()

plt.tight_layout()
plot_name = f"pi_test_exp{EXPERIMENT_ID}_SAT{SAT_COOL}_ZATpattern.png"
plt.savefig(plot_name, dpi=300, bbox_inches="tight")
plt.show()