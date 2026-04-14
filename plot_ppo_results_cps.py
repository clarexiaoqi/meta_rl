# plot_ppo_results_cps.py
# RAW-reward compatible plotting script
# - Subplot 1 uses episode_return_total / _energy / _comfort
# - Subplot 5 decomposition uses raw TotalEnergy_kWh and TempExceed_degC
# - Reheat signal plotted as % of max
# - Qsg and Qint added to Subplot 2 on a right y-axis (dual axis)
# - Exports a CSV with BOTH normalized obs columns and denormalized named state columns

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =============================================================================
# USER SETTINGS
# =============================================================================
BASE_DIR = r"C:\Users\jbak2\OneDrive - University of Nebraska\Desktop\CPS\Connect_Env_and_basic_RL\Apri_14"
EPISODE_CSV = os.path.join(BASE_DIR, "episode_rewards.csv")
LAST_LOG_CSV = os.path.join(BASE_DIR, "last_episode_log.csv")

DT_SECONDS = 1800.0
DT_HOURS = DT_SECONDS / 3600.0

DAYS_TO_PLOT = 3
STEPS_3DAYS = int((24 * DAYS_TO_PLOT) / DT_HOURS)

# State = [T_env, T_zone, T_cor, T_out, Qsg, Qint, hour_sin, hour_cos]
OBS_LOW = np.array([10., 15., 20., -40., 0., 50., -1., -1.], dtype=float)
OBS_HIGH = np.array([45., 28., 28., 40., 1100., 180., 1., 1.], dtype=float)
STATE_NAMES = ["T_env", "T_zone", "T_cor", "T_out", "Qsg", "Qint", "hour_sin", "hour_cos"]

# Comfort box
COMFORT_LB = 21.0
COMFORT_UB = 24.0
OCC_START = 7.0
OCC_END = 20.0

# Must match env alpha
ALPHA = 0.3

# Must match env
QH_REHEAT_MAX_W = 300.0
ETA_REHEAT = 0.9
OUTER_STEP_HR = DT_HOURS

OUT_PNG = os.path.join(BASE_DIR, "ppo_results_plots.png")
OUT_EXPORT_CSV = os.path.join(BASE_DIR, "last_episode_log_with_denorm_obs.csv")


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

obs_cols = [f"obs_{i}" for i in range(8)]

missing_obs_cols = [c for c in obs_cols if c not in last.columns]
if missing_obs_cols:
    raise ValueError(f"Missing observation columns in last log CSV: {missing_obs_cols}")

# =============================================================================
# EXPORT CSV WITH DENORMALIZED OBSERVATIONS
# =============================================================================
obs_all = last[obs_cols].to_numpy(dtype=float)
obs_all_denorm = denorm_obs(obs_all)

export_df = last.copy()

for i, name in enumerate(STATE_NAMES):
    export_df[name] = obs_all_denorm[:, i]

# Rename original normalized obs columns for readability
export_df = export_df.rename(columns={f"obs_{i}": f"{STATE_NAMES[i]}_norm" for i in range(8)})

export_df.to_csv(OUT_EXPORT_CSV, index=False)
print(f"Saved: {OUT_EXPORT_CSV}")

# =============================================================================
# EPISODE-LEVEL DATA
# =============================================================================
episodes = ep["episode"].to_numpy(dtype=int)
ep_total = safe_col(ep, "episode_return_total", fallback="episode_return")
ep_energy = safe_col(ep, "episode_return_energy")
ep_comfort = safe_col(ep, "episode_return_comfort")

# =============================================================================
# FIRST 3 DAYS SLICE
# =============================================================================
last3 = last.iloc[:STEPS_3DAYS].copy()
tstep3 = last3["tstep"].to_numpy(dtype=int)
x_hours_3 = tstep3 * DT_HOURS

obs3 = last3[obs_cols].to_numpy(dtype=float)
obs3_den = denorm_obs(obs3)

# =============================================================================
# DIAGNOSTIC CHECK: RAW AND NORMALIZED RANGES
# =============================================================================
print("\n===== OBSERVATION RANGE DIAGNOSTICS =====")
full_obs = last[obs_cols].to_numpy(dtype=float)
full_obs_den = denorm_obs(full_obs)

for i, name in enumerate(STATE_NAMES):
    norm_min = np.min(full_obs[:, i])
    norm_max = np.max(full_obs[:, i])
    raw_min = np.min(full_obs_den[:, i])
    raw_max = np.max(full_obs_den[:, i])

    print(
        f"{name:>9s} | "
        f"norm_min={norm_min:8.4f}, norm_max={norm_max:8.4f} | "
        f"raw_min={raw_min:10.4f}, raw_max={raw_max:10.4f} | "
        f"expected_raw=[{OBS_LOW[i]:.4f}, {OBS_HIGH[i]:.4f}]"
    )

print("\n===== OUT-OF-RANGE COUNTS =====")
for i, name in enumerate(STATE_NAMES):
    below_0 = np.sum(full_obs[:, i] < 0.0)
    above_1 = np.sum(full_obs[:, i] > 1.0)
    print(f"{name:>9s} | below_0={below_0:5d}, above_1={above_1:5d}")

if "T_env_raw" in last.columns and "T_env_norm" in last.columns:
    print("\n===== DIRECT T_env CHECK FROM LOGGED INFO =====")
    print(f"T_env_raw  min={last['T_env_raw'].min():.4f}, max={last['T_env_raw'].max():.4f}")
    print(f"T_env_norm min={last['T_env_norm'].min():.4f}, max={last['T_env_norm'].max():.4f}")
    print(f"T_env_norm out-of-range count = {np.sum((last['T_env_norm'] < 0.0) | (last['T_env_norm'] > 1.0))}")

# =============================================================================
# DENORMALIZED STATE VARIABLES FOR PLOTTING
# =============================================================================
T_env_3 = obs3_den[:, 0]
T_zone_3 = obs3_den[:, 1]
T_cor_3 = obs3_den[:, 2]
T_out_3 = obs3_den[:, 3]
Qsg_3 = obs3_den[:, 4]
Qint_3 = obs3_den[:, 5]
hour_sin_3 = obs3_den[:, 6]
hour_cos_3 = obs3_den[:, 7]

# Setpoints
SAT_3 = safe_col(last3, "SAT_sp", fallback="action_0")
ZAT_used_3 = safe_col(last3, "ZAT_sp_used")
if ZAT_used_3 is None:
    ZAT_used_3 = safe_col(last3, "ZAT_sp")
if ZAT_used_3 is None:
    ZAT_used_3 = safe_col(last3, "action_1")

# PI vars
m_fan_3 = safe_col(last3, "m_fan")
damper_sig_3 = safe_col(last3, "DamperSignal")

# Energy (kWh per step)
E_tot_3 = safe_col(last3, "TotalEnergy_kWh")
E_cool_3 = safe_col(last3, "CoolingEnergy_kWh")
E_heat_3 = safe_col(last3, "HeatingEnergy_kWh")
E_reheat_3 = safe_col(last3, "ReheatEnergy_kWh")
E_fan_3 = safe_col(last3, "FanEnergy_kWh")

# Reward decomposition
reward_3 = safe_col(last3, "Reward", fallback="reward")
temp_exceed_3 = safe_col(last3, "TempExceed_degC")

r_energy_3 = -E_tot_3 if E_tot_3 is not None else None
r_comfort_3 = -(ALPHA * temp_exceed_3) if temp_exceed_3 is not None else None

# Reheat % of max
reheat_pct_3 = None
if E_reheat_3 is not None:
    P_reheat_avg_kW = np.array(E_reheat_3, dtype=float) / max(OUTER_STEP_HR, 1e-9)
    Q_reheat_avg_kW = P_reheat_avg_kW * ETA_REHEAT
    Qh_max_kW = QH_REHEAT_MAX_W / 1000.0
    reheat_pct_3 = 100.0 * (Q_reheat_avg_kW / max(Qh_max_kW, 1e-9))
    reheat_pct_3 = np.clip(reheat_pct_3, 0.0, 100.0)

# =============================================================================
# PLOTTING
# =============================================================================
plt.figure(figsize=(14, 18))
gs = plt.GridSpec(5, 1, height_ratios=[1.25, 2.3, 1.7, 1.7, 1.9], hspace=0.35)

# -----------------------------
# Subplot 1
# -----------------------------
ax1 = plt.subplot(gs[0])
ax1.set_title("Subplot 1: Reward over Episodes (Total + Objectives)")
if ep_total is not None:
    ax1.plot(episodes, ep_total, linewidth=1.8, label="Episode Return (Total)")
if ep_energy is not None:
    ax1.plot(episodes, ep_energy, linewidth=1.4, linestyle=":", label="Episode Return (Energy)")
if ep_comfort is not None:
    ax1.plot(episodes, ep_comfort, linewidth=1.4, linestyle="-.", label="Episode Return (Comfort)")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Episode Return")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="best")

# -----------------------------
# Subplot 2
# -----------------------------
ax2 = plt.subplot(gs[1])
ax2.set_title("Subplot 2: First 3 Days - Temps/Setpoints + Qsg/Qint (Occupied Comfort Box)")

ax2.plot(x_hours_3, T_env_3, linewidth=1.8, linestyle=":", label="T_env")
ax2.plot(x_hours_3, T_zone_3, linewidth=2.0, label="T_zone")
ax2.plot(x_hours_3, T_cor_3, linewidth=1.6, linestyle="--", label="T_cor")
ax2.plot(x_hours_3, T_out_3, linewidth=1.6, linestyle="--", color="black", label="T_out")

if SAT_3 is not None:
    ax2.plot(x_hours_3, SAT_3, linewidth=1.6, linestyle="-.", label="SAT_sp (Action)")
if ZAT_used_3 is not None:
    ax2.plot(x_hours_3, ZAT_used_3, linewidth=1.6, color="red", label="ZAT_sp (Action)")

add_occupied_boxes(ax2, x_hours_3, COMFORT_LB, COMFORT_UB, OCC_START, OCC_END, alpha=0.15)

ax2.set_xlabel("Time (hours from episode start)")
ax2.set_ylabel("Temperature / Setpoints (°C)")
ax2.grid(True, alpha=0.3)

ax2b = ax2.twinx()
lines2b = []
labels2b = []

l_qsg, = ax2b.plot(x_hours_3, Qsg_3, linewidth=1.4, linestyle="--", color="green", label="Qsg (W)")
lines2b.append(l_qsg)
labels2b.append("Qsg (W)")

l_qint, = ax2b.plot(x_hours_3, Qint_3, linewidth=1.4, linestyle=":", color="purple", label="Qint (W)")
lines2b.append(l_qint)
labels2b.append("Qint (W)")

ax2b.set_ylabel("Gains (W)")

lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines2 + lines2b, labels2 + labels2b, loc="upper right")

# -----------------------------
# Subplot 3
# -----------------------------
ax3 = plt.subplot(gs[2])
ax3.set_title("Subplot 3: First 3 Days - PI Loop Variables + Reheat (%)")

if damper_sig_3 is not None:
    ax3.plot(x_hours_3, damper_sig_3, linewidth=1.8, label="Damper Signal (%)")

ax3.set_xlabel("Time (hours)")
ax3.set_ylabel("Damper (%)")
ax3.grid(True, alpha=0.3)

ax3b = ax3.twinx()
line_handles = []
line_labels = []

if m_fan_3 is not None:
    l_mfan, = ax3b.plot(x_hours_3, m_fan_3, linewidth=1.8, linestyle="--", label="m_fan (kg/s)")
    line_handles.append(l_mfan)
    line_labels.append("m_fan (kg/s)")
ax3b.set_ylabel("m_fan (kg/s)")

ax3c = ax3.twinx()
ax3c.spines["right"].set_position(("axes", 1.10))
ax3c.spines["right"].set_visible(True)

if reheat_pct_3 is not None:
    l_reheat, = ax3c.plot(x_hours_3, reheat_pct_3, linewidth=1.8, linestyle=":", label="Reheat (% of max)")
    ax3c.set_ylim(-5, 105)
    ax3c.set_ylabel("Reheat (%)")
    line_handles.append(l_reheat)
    line_labels.append("Reheat (% of max)")

lines3, labels3 = ax3.get_legend_handles_labels()
ax3.legend(lines3 + line_handles, labels3 + line_labels, loc="upper right")

# -----------------------------
# Subplot 4
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
# Subplot 5
# -----------------------------
ax5 = plt.subplot(gs[4])
ax5.set_title("Subplot 5: Reward over Timesteps (First 3 Days) + Decomposition (RAW)")

if reward_3 is not None:
    ax5.plot(x_hours_3, reward_3, linewidth=1.4, label="Reward (total) per step")
if r_energy_3 is not None:
    ax5.plot(x_hours_3, r_energy_3, linewidth=1.2, linestyle="--", label="-TotalEnergy_kWh (per step)")
if r_comfort_3 is not None:
    ax5.plot(x_hours_3, r_comfort_3, linewidth=1.2, linestyle=":", label=f"-{ALPHA}*TempExceed_degC (per step)")

ax5.set_xlabel("Time (hours)")
ax5.set_ylabel("Reward / components")
ax5.grid(True, alpha=0.3)
ax5.legend(loc="upper right")

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {OUT_PNG}")
plt.show()
