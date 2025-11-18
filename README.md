# Task1 Updated Environment

This repository contains updates of the RL environment `ContinuousBuildingControlEnvironment`. The environment originally modeled a single-zone building using a simplified direct-actuation approach.  

In **v4**, the environment introduces:

1.**A supervisory control layer for SAT and ZAT setpoints**
2.**A realistic inner PI control loop**
3.**A correct two-timescale thermal model using discretization**
4.**Updated York DNZ060 cooling performance curves**

Below is a clear comparison of **OG meta_rl vs v4**.

| Component | OG meta_rl Code | v4 Code | Impact |
|----------|---------|---------|--------|
| **HVAC actuation** | Direct heating/cooling power (`a_t` → W) | Damper → airflow → HVAC load | More realistic actuator physics |
| **Setpoints (SAT/ZAT)** | Not in control loop | SAT_sp, ZAT_sp implemented in supervisory layer (Fixed yet) | Ready for RL control in next version |
| **PI Controller** | **None** | Full PI controller with anti-windup | Enables supervisory RL |
| **Thermal model timestep** | Updated every **30 min** only | Updated every **5 min** inside PI loop | Captures fast HVAC and thermal dynamics |
| **Energy integration** | Uses 1800 s only | Integrates energy per 300 s PI substep | More accurate kWh calculation |
| **COP curves** | Simplified polynomial-based COP | Full York DNZ060 curves (CAPFT, EIRFT, CAPFFF, EIRFFF) | Higher fidelity cooling performance |
| **Fan model** | Same as OG | Preserved | No change |
