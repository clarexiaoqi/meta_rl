# EnergyPlus–Python FMU Control

This example runs an EnergyPlus FMU from Python and exchanges control inputs and simulation outputs every timestep.

## Main files

- `Test_Building.idf` – EnergyPlus model
- `Test_Building.fmu` – FMU exported from EnergyPlus
- `Control.py` – Python controller using PyFMI
- `runs/` – simulation results

## PyFMI

`Control.py` uses the **PyFMI** library:

```python
from pyfmi import load_fmu
```

Main PyFMI functions used:

| Function | Purpose |
|---|---|
| `load_fmu()` | Loads the EnergyPlus FMU |
| `setup_experiment()` | Sets simulation start/stop times |
| `enter_initialization_mode()` / `exit_initialization_mode()` | Initializes the FMU |
| `model.set()` | Sends control actions from Python to EnergyPlus |
| `model.do_step()` | Advances EnergyPlus one timestep |
| `model.get()` | Reads EnergyPlus outputs into Python |
| `model.terminate()` | Ends the FMU simulation |
| `model.free_instance()` | Releases the FMU instance |

## Two control actions

Python controls both:

```python
model.set("ahu_sat_sp_C", ahu_sat_target)
model.set("zone_temp_sp_C", zone_temp_target)
```

- `ahu_sat_sp_C` – AHU supply-air temperature setpoint
- `zone_temp_sp_C` – zone air temperature setpoint

The current `simple_controller()` is only an example for testing the real-time interaction. It calculates **both actions** from the latest zone temperature.

```python
ahu_sat_target, zone_temp_target = simple_controller(zone_temp)
```

A hotter zone gives lower SAT and ZAT targets; a colder zone gives higher targets.

## For Yizhong

The main part to replace is:

```python
def simple_controller(T_zone):
    ...
```

Replace this simple rule with your controller or RL policy. The two RL actions should then be assigned to:

```python
ahu_sat_target
zone_temp_target
```

The FMU communication can stay the same:

```text
Controller / RL
      |
      |  [SAT_sp, ZAT_sp]
      |  model.set()
      v
EnergyPlus FMU
      |
      |  model.get()
      v
Python
```

If FMU variable names are changed in the IDF, also update the names used in `model.set()` and `model.get()`.

Current timestep: **15 minutes (900 s)**  
EnergyPlus version: **22.2**
