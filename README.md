# Model Predictive Control (MPC) for HVAC Control

This project implements a Model Predictive Control (MPC) framework for HVAC control based on a 3R2C building thermal model.

The controller optimizes the Supply Air Temperature (SAT) and Zone Air Temperature (ZAT) setpoints over a finite prediction horizon while balancing energy consumption and indoor thermal comfort.

The framework includes

- 3R2C building thermal model
- Multi-step state prediction
- Receding horizon optimization
- PI controller simulation
- Offline performance evaluation

---

# Framework

The MPC workflow consists of four modules.

```
Prediction Environment
        ↓
Prediction Model
        ↓
MPC Controller
        ↓
Offline Evaluation
```

---

# Project Structure

```
Env_develop_mpc_draft.py
        ↓
mpc_model.py
        ↓
mpc_controller.py
        ↓
mpc_test.py
```

---

# Environment

```
Env_develop_mpc_draft.py
```

This module provides

- HVAC simulation environment
- 3R2C thermal dynamics
- PI controller
- reward computation
- state transition
- weather data interface

---

# Prediction Model

```
mpc_model.py
```

This module implements

- multi-step prediction
- state rollout
- objective evaluation
- energy computation
- comfort penalty

The prediction model is used by the optimizer to evaluate candidate control sequences.

---

# MPC Controller

```
mpc_controller.py
```

This module implements

- receding horizon MPC
- constrained optimization
- SLSQP solver
- warm-start optimization
- SAT and ZAT optimization

At every control step, the controller solves an optimization problem and applies only the first control action.

---

# Offline Evaluation

Run

```bash
python mpc_test.py
```

The evaluation generates

- indoor temperature profile
- SAT trajectory
- ZAT trajectory
- outdoor temperature
- solar heat gain
- energy consumption
- comfort violations
- optimization statistics

Example result

<p align="center">
<img src="1.png" width="900">
</p>

---

# Performance

Typical evaluation metrics

| Metric | Value |
|--------|------:|
| Total Return | -2.2940 |
| Energy Consumption | 137.63 kWh |
| Violation Hours | 2.50 h |
| Temperature Exceedance | 0.00284 °C·hr |
| Solver Success Rate | 100% |
| Average Solver Time | 0.038 s |

These results demonstrate that the MPC controller successfully maintains indoor comfort while achieving energy-efficient HVAC control.

---

# Main Files

```
Env_develop_mpc_draft.py

mpc_model.py

mpc_controller.py

mpc_test.py
```

---

# Current Status

The current implementation supports

- exact 3R2C building model
- finite-horizon optimization
- online receding horizon control
- offline performance evaluation

Future work includes

- system parameter identification
- weather prediction
- robustness evaluation
- comparison with learning-based controllers

---

# Repository

MPC Development Branch

```
mpc
```