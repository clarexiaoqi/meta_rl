# Task1 Updated Environment

This repository contains updates of the RL environment `ContinuousBuildingControlEnvironment`.
The environment originally modeled a single-zone building using a simplified direct-actuation approach.  
In **v4**, the environment introduces:

1.**A supervisory control layer for SAT and ZAT setpoints**
2. **A realistic inner PI control loop**
3. **A correct two-timescale thermal model using discretization**
4. **Updated York DNZ060 cooling performance curves**

Below is a clear comparison of **OG vs v4**.

| Component | OG Code | v4 Code | Impact |
|----------|---------|---------|--------|
| **HVAC actuation** | Direct heating/cooling power (`a_t` → W) | Damper → airflow → HVAC load | More realistic actuator physics |
| **Thermal model timestep** | Updated every **30 min** only | Updated every **5 min** inside PI loop | Captures fast HVAC and thermal dynamics |
| **PI Controller** | **None** | Full PI controller with anti-windup | Enables supervisory RL |
| **Setpoints (SAT/ZAT)** | Fixed; not in control loop | SAT_sp, ZAT_sp implemented in supervisory layer | Ready for RL control in next version |
| **COP curves** | Simplified polynomial-based COP | Full York DNZ060 curves (CAPFT, EIRFT, CAPFFF, EIRFFF) | Higher fidelity cooling performance |
| **Fan model** | Same as OG | Preserved | No change |
| **Reward** | -energy + utility | -energy + utility | Preserved with consistent structure |
| **Energy integration** | Uses 1800 s only | Integrates energy per 300 s PI substep | More accurate kWh calculation |
| **State evolution** | Single-step linear model | Multi-step (6×) PI-integrated linear model | Better thermal realism |
| **Action meaning** | HVAC power input (W) | Will become SAT_sp & ZAT_sp actions | Matches building automation practice |
