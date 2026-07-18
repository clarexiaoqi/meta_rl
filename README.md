# Online Adaptive MPC for HVAC Control

This repository implements an **Online Adaptive Model Predictive Control (MPC)** framework for HVAC systems with **online parameter identification** and **weather forecast uncertainty**.

Unlike conventional MPC, which assumes an accurate building model is already available, this framework starts from a generic thermal model and continuously estimates unknown building parameters from observed historical data. The identified model is then used to update the prediction model online, allowing the controller to adapt to buildings with initially unknown thermal characteristics.

---

# Highlights

- Online parameter identification using historical observations
- Adaptive prediction model updated during operation
- Online MPC with periodically identified models
- Weather forecast uncertainty with temporally correlated forecast errors
- Easily extensible to different buildings and HVAC systems

---

# Overall Workflow

```
Historical Measurements
          │
          ▼
Collect Identification Data
          │
          ▼
Online Parameter Identification
          │
          ▼
Update Prediction Model
          │
          ▼
Online Adaptive MPC
          │
          ▼
Apply Control to Real Environment
```

The real environment is **never modified**.

Only the prediction model used by MPC is updated after parameter identification.

---

# Weather Forecast Uncertainty

Real HVAC systems never have perfect future weather information.

Instead of using perfect weather predictions, forecast uncertainty is introduced into the MPC prediction model.

Outdoor air temperature is modeled as

```
T̂_out = T_out + e_T
```

Solar heat gain is modeled as

```
Q̂_sg = Q_sg (1 + e_Q)
```

Forecast errors follow an AR(1) process

```
e_k = ρ e_(k−1) + √(1−ρ²) ε_k
```

where

- Temperature forecast standard deviation: **0.8°C**
- Solar forecast relative standard deviation: **20%**
- Temporal correlation coefficient: **ρ = 0.8**

A new forecast trajectory is generated before every MPC optimization and remains fixed throughout that optimization.

This setup better reflects practical HVAC deployment where future weather information is uncertain.

---

# Repository Structure

```
.
├── data/                                   # Weather data
├── plots/                                  # Figures
├── results/                                # Experimental results
├── Env_develop_mpc_draft.py                # HVAC environment
├── mpc_controller.py                       # MPC controller
├── mpc_model.py                            # Prediction model
├── parameter_identification.py             # Online parameter identification
├── parameter_identification_model.py       # Thermal model
├── mpc_test.py                             # Oracle MPC
├── mpc_test_online.py                      # Online adaptive MPC
├── collect_identification_data.py          # Data collection
├── parameter_config.py                     # Parameters
└── README.md
```

---

# Experimental Results

## Online Adaptive MPC

The figure below shows the control performance of the proposed online adaptive MPC framework.

![Online Adaptive MPC](1.png)

The controller successfully adapts the prediction model after online parameter identification and maintains stable HVAC control performance.

---

## Online Parameter Identification

The identified thermal parameters gradually converge toward the true building parameters, improving prediction accuracy for MPC optimization.

![Online Parameter Identification](2.png)

---

# Running the Code

## Oracle MPC

```bash
python mpc_test.py
```

## Online Adaptive MPC

```bash
python mpc_test_online.py
```

---

# Output Files

Simulation outputs are automatically saved under

```
results/
```

including

- MPC control trajectories
- Energy consumption
- Indoor temperature
- Identified parameters
- Prediction accuracy
- Optimization history

Figures are saved under

```
plots/
```

---

# Main Features

Compared with conventional MPC, this framework provides

- Online thermal parameter estimation
- Adaptive prediction model update
- Robust control under weather forecast uncertainty
- Modular implementation for future extensions

---

# Future Work

Possible future extensions include

- Multi-zone buildings
- Nonlinear thermal dynamics
- Occupancy prediction
- Integration with reinforcement learning
- Adaptive prediction horizon

---

# License

This repository is intended for academic research purposes.
