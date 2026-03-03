# plot_ppo_results_cps.py
# Modified per your requests:
# 1) No rolling lines in subplot 1
# 2) No dual axis for SAT in subplot 2
# 3) No comfort LB/UB lines in subplot 2 (yellow box only)
# 4) T_out line is black
# 5) Plot ZAT_sp_used instead of HeatSP/CoolSP in subplot 2
# 6) Subplot 3: only DamperSignal and m_fan (no DamperEff, no COPc)
# 7) Subplot 5 x-axis is first 3 days (not full episode)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =============================================================================
# USER SETTINGS
# =============================================================================
BASE_DIR = r"C:\Users\jbak2\OneDrive - University of Nebraska\Desktop\CPS\Connect_Env_and_basic_RL\Mar_2"
EPISODE_CSV = os.path.join(BASE_DIR, "episode_rewards.csv")
LAST_LOG_CSV = os.path.join(BASE_DIR, "last_episode_log.csv")

DT_SECONDS = 1800.0
DT_HOURS = DT_SECONDS / 3600.0

DAYS_TO_PLOT = 3
STEPS_3DAYS = int((24 * DAYS_TO_PLOT) / DT_HOURS)  # 72hr / 0.5hr = 144

# Observation normalization bounds used in env:
OBS_LOW = np.array([10., 15., 20., -40., 0., 50., 0.], dtype=float)
OBS_HIGH = np.array([35., 28., 28.,  40., 1100., 180., 23.], dtype=float)

# Comfort band just for yellow box
COMFORT_LB = 21.0
COMFORT_UB = 24.0
OCC_START = 7.0
OCC_END = 20.0

ALPHA = 1.0
OUT_PNG = os.path.join(BASE_DIR, "ppo_results_plots.png")


# =============================================================================
# HELPERS
# =============================================================================
def denorm_obs(obs_norm: np.ndarray) -> np.ndarray:
    return obs_norm * (OBS_HIGH - OBS_LOW) + OBS_LOW


def add_occupied_boxes(ax, x_hours, y_low, y_high, occ_start=7.0, occ_end=20.0, alpha=0.12):
    max_hour = float(np.max(x_hours))
    n_days = int(np.floor(max_hour / 24.0)) + 1
    for d in range(n_days):
        x0 = d * 24.0 + occ_start
        x1 = d * 24.0 + occ_end
        if x1 < np.min(x_hours) or x0 > np.max(x_hours):
            continue
        rect = Rectangle(
            (x0, y_low),
            width=(x1 - x0),
            height=(y_high - y_low),
            facecolor="yellow",
            edgecolor=None,
            alpha=alpha,
            zorder=0,
        )
        ax.add_patch(rect)


def safe_col(df: pd.DataFrame, name: str, fallback: str = None):
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    if fallback and fallback in df.columns:
        return df[fallback].to_numpy(dtype=float)
    return None


# =============================================================================
# LOAD DATA
# =============================================================================
ep = pd.read_csv(EPISODE_CSV)
last = pd.read_csv(LAST_LOG_CSV)

# Episode series
episodes = ep["episode"].to_numpy(dtype=int)
ep_total = safe_col(ep, "episode_return_total", fallback="episode_return")
ep_energy = safe_col(ep, "episode_return_energy")
ep_comfort = safe_col(ep, "episode_return_comfort")

# Last episode time-series (full)
tstep = last["tstep"].to_numpy(dtype=int)
x_hours_full = tstep * DT_HOURS

# Slice first 3 days for all time-series subplots (2~5)
last3 = last.iloc[:STEPS_3DAYS].copy()
tstep3 = last3["tstep"].to_numpy(dtype=int)
x_hours_3 = tstep3 * DT_HOURS

# Denormalize obs for 3 days
obs_cols = [f"obs_{i}" for i in range(7)]
obs3 = last3[obs_cols].to_numpy(dtype=float)
obs3_den = denorm_obs(obs3)

T_zone_3 = obs3_den[:, 1]
T_out_3 = obs3_den[:, 3]

# Setpoints / actions for 3 days
SAT_3 = safe_col(last3, "SAT_sp", fallback="action_0")
ZAT_used_3 = safe_col(last3, "ZAT_sp_used")

# PI loop variables (3 days)
m_fan_3 = safe_col(last3, "m_fan")
damper_sig_3 = safe_col(last3, "DamperSignal")

# Energy variables (3 days)
E_tot_3 = safe_col(last3, "TotalEnergy_kWh")
E_cool_3 = safe_col(last3, "CoolingEnergy_kWh")
E_heat_3 = safe_col(last3, "HeatingEnergy_kWh")
E_reheat_3 = safe_col(last3, "ReheatEnergy_kWh")
E_fan_3 = safe_col(last3, "FanEnergy_kWh")

# Reward decomposition (3 days only for subplot 5)
reward_3 = safe_col(last3, "Reward", fallback="reward")
en_norm_3 = safe_col(last3, "EnergyNorm")
co_norm_3 = safe_col(last3, "ComfortNorm")

cum_reward_3 = np.cumsum(reward_3) if reward_3 is not None else None
r_energy_3 = -en_norm_3 if en_norm_3 is not None else None
r_comfort_3 = -(ALPHA * co_norm_3) if co_norm_3 is not None else None


# =============================================================================
# PLOTTING
# =============================================================================
plt.figure(figsize=(14, 18))
gs = plt.GridSpec(5, 1, height_ratios=[1.25, 2.3, 1.6, 1.7, 1.9], hspace=0.35)

# -----------------------------
# Subplot 1: Total reward over episodes + objectives (NO rolling)
# -----------------------------
ax1 = plt.subplot(gs[0])
ax1.set_title("Subplot 1: Reward over Episodes (Total + Objectives)")

if ep_total is not None:
    ax1.plot(episodes, ep_total, linewidth=1.8, label="Episode Return (Total)")
if ep_energy is not None:
    ax1.plot(episodes, ep_energy, linewidth=1.4, linestyle=":", label="Episode Return (Energy Component)")
if ep_comfort is not None:
    ax1.plot(episodes, ep_comfort, linewidth=1.4, linestyle="-.", label="Episode Return (Comfort Component)")

ax1.set_xlabel("Episode")
ax1.set_ylabel("Episode Return")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="best")

# -----------------------------
# Subplot 2: First 3 days temperatures + setpoints (NO dual axis, NO LB/UB lines)
# -----------------------------
ax2 = plt.subplot(gs[1])
ax2.set_title("Subplot 2: First 3 Days - Temperatures + Setpoints (Occupied Comfort Box)")

ax2.plot(x_hours_3, T_zone_3, linewidth=2.0, label="Zone Air Temp (T_zone)")
ax2.plot(x_hours_3, T_out_3, linewidth=1.6, linestyle="--", color="black", label="Outdoor Air Temp (T_out)")

if SAT_3 is not None:
    ax2.plot(x_hours_3, SAT_3, linewidth=1.6, linestyle="-.", label="SAT_sp (action)")

if ZAT_used_3 is not None:
    ax2.plot(x_hours_3, ZAT_used_3, linewidth=1.6, label="ZAT_sp_used")

add_occupied_boxes(ax2, x_hours_3, COMFORT_LB, COMFORT_UB, OCC_START, OCC_END, alpha=0.15)

ax2.set_xlabel("Time (hours from episode start)")
ax2.set_ylabel("Temperature / Setpoints (°C)")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

# -----------------------------
# Subplot 3: First 3 days PI-loop variables (ONLY DamperSignal and m_fan)
# -----------------------------
ax3 = plt.subplot(gs[2])
ax3.set_title("Subplot 3: First 3 Days - PI Loop Variables")

if damper_sig_3 is not None:
    ax3.plot(x_hours_3, damper_sig_3, linewidth=1.8, label="Damper Signal (%)")

ax3.set_xlabel("Time (hours)")
ax3.set_ylabel("Damper (%)")
ax3.grid(True, alpha=0.3)

ax3b = ax3.twinx()
if m_fan_3 is not None:
    ax3b.plot(x_hours_3, m_fan_3, linewidth=1.8, linestyle="--", label="m_fan (kg/s)")
ax3b.set_ylabel("m_fan (kg/s)")

lines3, labels3 = ax3.get_legend_handles_labels()
lines3b, labels3b = ax3b.get_legend_handles_labels()
ax3.legend(lines3 + lines3b, labels3 + labels3b, loc="upper right")

# -----------------------------
# Subplot 4: First 3 days energy breakdown
# -----------------------------
ax4 = plt.subplot(gs[3])
ax4.set_title("Subplot 4: First 3 Days - Energy per Step (kWh per 30-min)")

if E_tot_3 is not None:
    ax4.plot(x_hours_3, E_tot_3, linewidth=2.0, label="Total Energy (kWh)")
if E_cool_3 is not None:
    ax4.plot(x_hours_3, E_cool_3, linewidth=1.4, label="Cooling Energy (kWh)")
if E_heat_3 is not None:
    ax4.plot(x_hours_3, E_heat_3, linewidth=1.4, label="Heating Energy (kWh)")
if E_reheat_3 is not None:
    ax4.plot(x_hours_3, E_reheat_3, linewidth=1.4, label="Reheat Energy (kWh)")
if E_fan_3 is not None:
    ax4.plot(x_hours_3, E_fan_3, linewidth=1.4, label="Fan Energy (kWh)")

ax4.set_xlabel("Time (hours)")
ax4.set_ylabel("Energy (kWh/step)")
ax4.grid(True, alpha=0.3)
ax4.legend(loc="upper right")

# -----------------------------
# Subplot 5: Reward over timesteps (FIRST 3 DAYS) + decomposition
# (No cumulative reward line / no secondary axis)
# -----------------------------
ax5 = plt.subplot(gs[4])
ax5.set_title("Subplot 5: Reward over Timesteps (First 3 Days) + Decomposition")

if reward_3 is not None:
    ax5.plot(x_hours_3, reward_3, linewidth=1.4, label="Reward (total) per step")
if r_energy_3 is not None:
    ax5.plot(x_hours_3, r_energy_3, linewidth=1.2, linestyle="--", label="-EnergyNorm (per step)")
if r_comfort_3 is not None:
    ax5.plot(x_hours_3, r_comfort_3, linewidth=1.2, linestyle=":", label=f"-ComfortNorm (per step)")

ax5.set_xlabel("Time (hours)")
ax5.set_ylabel("Reward / components (dimensionless)")
ax5.grid(True, alpha=0.3)
ax5.legend(loc="upper right")

# Save + show
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {OUT_PNG}")
plt.show()