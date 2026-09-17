import csv
import os
from datetime import datetime
from pathlib import Path

from pyfmi import load_fmu


# ============================================================
# 1. Paths and simulation settings
# ============================================================

HERE = Path(__file__).resolve().parent
FMU = HERE / "Test_Building.fmu"

EP_DIR = r"C:\EnergyPlusV22-2-0"

START_DAY = 184   # July 3 in a non-leap year
DAYS = 1
DT = 900          # 15 minutes; matches IDF Timestep,4

# Make EnergyPlus 22.2 available to the FMU.
os.environ["PATH"] = EP_DIR + os.pathsep + os.environ.get("PATH", "")

if not FMU.exists():
    raise FileNotFoundError(f"FMU not found: {FMU}")


# ============================================================
# 2. Load the FMI 2.0 Co-Simulation FMU
# ============================================================

model = load_fmu(str(FMU), kind="CS")

start = (START_DAY - 1) * 86400
stop = start + DAYS * 86400

model.setup_experiment(
    start_time=start,
    stop_time_defined=True,
    stop_time=stop,
)
model.enter_initialization_mode()
model.exit_initialization_mode()


# ============================================================
# 3. Prepare result file
# ============================================================

run_folder = HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder.mkdir(parents=True, exist_ok=True)

POWER_NAMES = [
    "P_cooling_W",
    "P_ahu_heating_W",
    "P_reheat_W",
    "P_supply_fan_W",
    "P_return_fan_W",
]

OUTPUT_NAMES = [
    "T_outdoor_C",
    "T_ahu_supply_C",
    "ahu_sat_sp_read_C",
] + POWER_NAMES

zone_temp = None

# Python-controlled zone thermostat target.
# Keep it fixed here for this simple demonstration; an RL controller can
# replace this value with a time-varying action later.
ZONE_TEMP_SETPOINT = 24.0


# ============================================================
# 4. Real-time interaction loop
# ============================================================

try:
    with (run_folder / "results.csv").open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "time_end_s",
                "T_zone_used_C",
                "ahu_sat_sp_C",
                "zone_temp_sp_C",
                "T_zone_C",
            ]
            + OUTPUT_NAMES
            + [
                "P_HVAC_W",
                "E_HVAC_kWh_step",
            ]
        )

        for t in range(start, stop, DT):

            # ------------------------------------------------
            # A. Python decides the two control inputs
            # ------------------------------------------------
            zone_temp_target = ZONE_TEMP_SETPOINT

            # One AHU supply-air-temperature (SAT) target is used by both
            # the AHU cooling and central heating coil control.
            if zone_temp is None or zone_temp >= zone_temp_target:
                ahu_sat_target = 12.8
            else:
                ahu_sat_target = 18.0

            # ------------------------------------------------
            # B. Python -> EnergyPlus
            # ------------------------------------------------
            model.set("ahu_sat_sp_C", ahu_sat_target)
            model.set("zone_temp_sp_C", zone_temp_target)

            # ------------------------------------------------
            # C. EnergyPlus advances one 15-minute step
            # ------------------------------------------------
            status = model.do_step(
                current_t=t,
                step_size=DT,
                new_step=True,
            )

            if status not in (0, 1):
                raise RuntimeError(
                    f"FMU step failed at {t} s. FMI status = {status}"
                )

            # ------------------------------------------------
            # D. EnergyPlus -> Python
            # ------------------------------------------------
            previous_zone_temp = zone_temp
            zone_temp = float(model.get("T_zone_C")[0])

            # Extra outputs are recorded only for validation.
            values = [float(model.get(name)[0]) for name in OUTPUT_NAMES]
            hvac_power = sum(values[-len(POWER_NAMES):])
            hvac_energy = hvac_power * DT / 3_600_000

            writer.writerow(
                [
                    t + DT,
                    previous_zone_temp,
                    ahu_sat_target,
                    zone_temp_target,
                    zone_temp,
                ]
                + values
                + [
                    hvac_power,
                    hvac_energy,
                ]
            )

            file.flush()

            print(
                f"{t + DT:>8} s | "
                f"AHU SAT target = {ahu_sat_target:4.1f} C | "
                f"Zone SP = {zone_temp_target:4.1f} C | "
                f"Zone temperature = {zone_temp:5.2f} C"
            )

finally:
    try:
        model.terminate()
    finally:
        model.free_instance()


print()
print(f"Simulation complete.")
print(f"Results: {run_folder / 'results.csv'}")
