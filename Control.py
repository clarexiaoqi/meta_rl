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


def simple_controller(T_zone):
    """
    Simple two-action demonstration controller.

    Both AHU SAT and zone-air setpoint are Python actions.
    A hotter zone produces lower SAT and ZAT targets.
    A colder zone produces higher SAT and ZAT targets.

    Replace this function with the RL controller later.
    """
    if T_zone is None:
        # Initial actions before the first EnergyPlus response is available.
        return 15.25, 23.0

    # AHU SAT action: 17.7 C at/below 22 C, decreasing to 12.8 C at/above 24 C.
    ahu_sat_target = 17.7 - 2.45 * (T_zone - 22.0)
    ahu_sat_target = max(12.8, min(17.7, ahu_sat_target))

    # Zone setpoint action: 23.5 C at 22 C zone temperature,
    # 23.0 C at 23 C, and 22.5 C at 24 C.
    zone_temp_target = 23.0 - 0.5 * (T_zone - 23.0)
    zone_temp_target = max(22.0, min(24.0, zone_temp_target))

    return ahu_sat_target, zone_temp_target


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
            # A. Python decides BOTH control actions
            # ------------------------------------------------
            ahu_sat_target, zone_temp_target = simple_controller(zone_temp)

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
