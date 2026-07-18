# Model Predictive Control for HVAC Control

This folder contains two MPC implementations for the same single-zone HVAC environment:

1. **Primary MPC** using the known 3R2C thermal model.
2. **Online adaptive MPC** starting from a generic 3R2C model, estimating the unknown thermal parameters from observed transitions, and continuing control with the identified model.

Both controllers optimize the Supply Air Temperature (SAT) and Zone Air Temperature (ZAT) setpoints over a finite prediction horizon. The objective balances HVAC energy consumption and indoor thermal comfort.

---

## Main Features

- Single-zone 3R2C building thermal model
- SAT and ZAT setpoint optimization
- Receding-horizon MPC using SLSQP
- Warm-started optimization
- Internal PI controller for HVAC actuation
- Daytime and nighttime comfort constraints
- Online identification of five 3R2C parameters
- Outdoor-temperature and solar-gain forecast uncertainty
- CSV result logging and figure generation
- Solver timing, convergence, and fallback statistics

---

## Project Structure

```text
mpc/
├── Env_develop_mpc_draft.py
├── mpc_model.py
├── mpc_controller.py
├── mpc_test.py
├── mpc_test_online.py
├── parameter_config.py
├── parameter_identification_model.py
├── parameter_identification.py
├── collect_identification_data.py
├── data/
│   ├── weather_data_2013_to_2017_summer_pandas.csv
│   ├── weather_data_2013_to_2017_winter_pandas.csv
│   └── weather_data_2017_pandas.csv
├── results/
└── plots/
```

### Main files

- **`Env_develop_mpc_draft.py`**  
  Defines the HVAC simulation environment, 3R2C thermal dynamics, PI controller, energy model, comfort bounds, weather interface, and prediction interface.

- **`mpc_model.py`**  
  Performs multi-step prediction for candidate SAT and ZAT sequences. It also contains the wrapper used to switch from the generic prediction model to the identified model.

- **`mpc_controller.py`**  
  Implements receding-horizon MPC with SciPy SLSQP. At each control step, it optimizes a sequence of future actions and applies only the first action.

- **`mpc_test.py`**  
  Runs the Primary MPC baseline using the known 3R2C parameters.

- **`mpc_test_online.py`**  
  Runs online adaptive MPC with one-time parameter identification and noisy weather forecasts.

- **`collect_identification_data.py`**  
  Collects transition data for parameter-identification tests.

- **`parameter_identification.py`** and **`parameter_identification_model.py`**  
  Implement the nonlinear least-squares identification of the five 3R2C parameters.

---

## MPC Formulation

At each control step, the controller solves

\[
\min_{\mathbf{u}_{0:H-1}}
\sum_{k=0}^{H-1}
\left(
J_{\mathrm{energy},k}
+
J_{\mathrm{comfort},k}
\right),
\]

subject to the 3R2C state transition and the SAT/ZAT bounds.

The control input is

\[
u_k = [SAT_k,\; ZAT_k].
\]

Only the first optimized action is applied to the real environment. The remaining sequence is shifted and used as the warm start for the next MPC optimization.

With the default 30-minute time step and a horizon of 6 steps, the prediction window is 3 hours.

---

## Primary MPC

The Primary MPC uses the known 3R2C parameters in both the real environment and the MPC prediction model.

Run:

```bash
python mpc_test.py
```

This experiment provides an optimization-based reference under ideal model and disturbance assumptions.

Typical output files are saved under:

```text
results/mpc/
plots/mpc/
```

---

## Online Adaptive MPC

The online adaptive MPC separates the real environment from the MPC prediction environment.

### Real environment

- Always uses the hidden true 3R2C parameters.
- Uses the actual weather trajectory from the dataset.
- Is never modified by parameter identification.

### Prediction environment

- Starts from generic 3R2C parameters.
- Uses only previously observed transitions for parameter identification.
- Replaces the generic prediction model with the identified model after the selected identification period.

The default identification period is 24 hours, corresponding to 48 transitions with a 30-minute time step.

Run:

```bash
python mpc_test_online.py
```

Run without opening the plot window:

```bash
python mpc_test_online.py --no_plot
```

Example with custom settings:

```bash
python mpc_test_online.py \
  --start 17664 \
  --end 19872.5 \
  --horizon 6 \
  --identification_hours 24 \
  --forecast_temperature_std 0.8 \
  --forecast_solar_relative_std 0.20 \
  --forecast_error_correlation 0.8 \
  --forecast_seed 35 \
  --no_plot
```

---

## Weather Forecast Uncertainty

The online MPC does not use perfect future weather values directly. Forecast uncertainty is added to two variables.

### Outdoor air temperature

Additive Gaussian error is used:

\[
\hat{T}_{out,k}=T_{out,k}+e_{T,k}.
\]

The default marginal standard deviation is

```text
0.8 °C
```

### Solar heat gain

Multiplicative Gaussian error is used:

\[
\hat{Q}_{sg,k}=Q_{sg,k}(1+e_{Q,k}).
\]

The default relative standard deviation is

```text
20%
```

Negative solar-gain forecasts are clipped to zero.

### Temporal correlation

Forecast errors follow an AR(1) process:

\[
e_k=\rho e_{k-1}+\sqrt{1-\rho^2}\varepsilon_k.
\]

The default correlation coefficient is

```text
rho = 0.8
```

One noisy forecast trajectory is generated before each MPC optimization and remains fixed during that optimization. A new forecast is generated at the next real control step.

The controller receives future weather forecasts, but it does not receive future indoor temperatures, future state transitions, future rewards, or the hidden true parameters.

---

## Main Command-Line Options

| Option | Default | Description |
|---|---:|---|
| `--start` | `17664.0` | Simulation start time in hours |
| `--end` | `19872.5` | Simulation end time in hours |
| `--horizon` | `6` | MPC prediction horizon in steps |
| `--maxiter` | `100` | Maximum SLSQP iterations per solve |
| `--ftol` | `3.37e-6` | SLSQP function tolerance |
| `--default_sat` | `14.5` | Initial/fallback SAT setpoint |
| `--default_zat` | `23.0` | Initial/fallback ZAT setpoint |
| `--identification_hours` | `24.0` | Data-collection period before identification |
| `--identification_max_nfev` | `500` | Maximum least-squares evaluations |
| `--forecast_temperature_std` | `0.8` | Temperature forecast error standard deviation in °C |
| `--forecast_solar_relative_std` | `0.20` | Relative solar forecast error standard deviation |
| `--forecast_error_correlation` | `0.8` | AR(1) correlation coefficient |
| `--forecast_seed` | `35` | Forecast-error random seed |
| `--no_plot` | off | Save figures without opening a window |

To display all options:

```bash
python mpc_test_online.py --help
```

---

## Output Files

The online adaptive MPC saves results under:

```text
results/online_mpc/
plots/online_mpc/
```

Main output files:

- **`online_mpc_results_profile.csv`**  
  Full control trajectory, thermal states, actions, disturbances, weather forecasts, comfort metrics, energy metrics, solver statistics, and prediction-model parameters.

- **`online_mpc_results_summary.csv`**  
  One-row summary containing total return, energy use, comfort violations, solver performance, forecast settings, and parameter-identification results.

- **`online_identified_parameters.json`**  
  Generic, identified, and true parameters for post-experiment comparison, together with relative identification errors and weather-forecast settings.

- **`online_mpc_control_profile.png`** and **`.pdf`**  
  Indoor temperature, comfort bounds, SAT, ZAT, outdoor temperature, and solar heat gain.

---

## Dependencies

The code requires Python 3 and the following packages:

```text
numpy
pandas
scipy
matplotlib
gym
```

A simple installation command is:

```bash
pip install numpy pandas scipy matplotlib gym
```

---

## Notes

- Run the scripts from the project folder so that the local Python modules and weather files can be found correctly.
- The weather CSV files must remain inside the `data/` folder.
- The AR(1) correlation parameter must satisfy

```text
0 <= rho < 1
```

- Changing only the default values inside `run_online_mpc_test()` does not change a normal command-line run if `parse_args()` passes different values. For normal script execution, change the command-line arguments or the defaults in `parse_args()`.

---

## Current Status

The current implementation supports:

- Primary MPC with known parameters
- Online adaptive MPC with generic initial parameters
- One-time nonlinear least-squares parameter identification
- Correlated weather forecast uncertainty
- Closed-loop SAT/ZAT control
- Full trajectory logging and solver diagnostics
- Direct comparison with learning-based HVAC controllers

Possible future extensions include repeated parameter updates, noisy sensor measurements, multi-zone building models, alternative weather-forecast models, and broader robustness experiments.
