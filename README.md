# DNZ060 Cooling Model Update (PI-Control Environment)

## Updated Files:
1. continuous_building_environment.py
2. test_pi_cooling.py
No other files are needed — these two are all you need to run the test.

## Objective:
This repository contains updates to the HVAC cooling environment used for our Reinforcement Learning (RL)–based building control project. Currently, we are validating and stabilizing the PI-controlled cooling loop, and this update focuses on improving the cooling COP model using real RTU performance curves.

1. Added York Affinity DNZ060 Performance Curves

We integrated the manufacturer cooling performance curves for the York Affinity DNZ060 rooftop unit:
	•	CAPFT – Cooling capacity modifier vs temperature
	•	EIRFT – Energy input ratio modifier vs temperature
	•	CAPFFF – Capacity modifier vs flow fraction
	•	EIRFFF – Energy input ratio modifier vs flow fraction

  COP_c = (COP_rated * CAPFT * CAPFFF) / (EIRFT * EIRFFF)

  The curves depend on:
	•	Outdoor dry-bulb temperature
	•	Supply air temperature
	•	Fan flow fraction (m_fan / m_design)

2. Capacity Scaling (1/3)

The DNZ060 RTU is sized to serve three thermal zones, but our simulation environment models only one zone. Therefore, we scale the coil capacity by: capacity_scale = 1/3

3. Current Control Mode (Fixed ZAT & SAT)

At this stage, RL actions are not active yet.

We are still validating the physics of the PI control loop.
	•	ZAT_sp is fixed at 24°C (occupied) / 28°C (unoccupied)
	•	SAT_sp is fixed (constant supply air temperature; no reset strategy)
	•	Only the PI damper control is operating
	•	This allows us to verify:
	•	Correct COP behavior
	•	Proper PI response
	•	Realistic energy and airflow values

Later, the RL agent will control ZAT_sp and SAT_sp.
