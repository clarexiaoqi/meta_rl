# Basic PPO-Based HVAC Control for a Simplified Building CPS Environment

This repository contains a new environment and a PPO training pipeline for continuous HVAC control in a simplified building. The agent learns two continuous control actions:

- **Supply air temperature setpoint (SAT)**
- **Zone air temperature setpoint (ZAT)**

The objective is to reduce HVAC energy use while maintaining indoor thermal comfort.


# Files

•	Env_develop.py
Custom OpenAI Gym environment: ContinuousBuildingControlEnvironment
•	train_ppo_cps.py
PPO training script implemented in PyTorch
•	plot_ppo_results_cps.py
Plotting script for training results and final episode behavior
•	episode_rewards.csv
Episode-level reward and objective summaries generated after training
•	last_episode_log.csv
Timestep-level log of the final rollout generated after training
•	data/
Input weather/internal gains data CSV used by the environment
•	ppo_runs_cps/
Saved PPO checkpoints

# Environment

• Class

ContinuousBuildingControlEnvironment(gym.Env)

• Action Space

Continuous 2D Box:
	•	SAT_sp ∈ [10.0, 15.5] °C
	•	ZAT_sp ∈ [18.0, 26.0] °C

• Observation Space

Normalized 7-dimensional state:
	1.	T_env
	2.	T_zone
	3.	T_cor
	4.	T_out
	5.	Qsg
	6.	Qint
	7.	Hour

The environment internally normalizes the state to [0, 1].

• Reward

Raw reward at each step:

r_t = - \left( \text{TotalEnergy}_{kWh} + \alpha \cdot \text{TempExceed}_{^\circ C} \right)

where:
	•	TotalEnergy_kWh = cooling + heating + reheat + fan energy
	•	TempExceed_degC = linear comfort violation outside the comfort band
	•	comfort is only penalized during occupied hours (7:00 to 20:00)

• Time Resolution
	•	Main RL timestep: 30 minutes
	•	Internal PI control timestep: 5 minutes


# PPO Training

The training script uses a custom PPO implementation in PyTorch with:
	•	tanh-squashed Gaussian policy
	•	clipped surrogate objective
	•	generalized advantage estimation (GAE)
	•	learning rate annealing
	•	entropy bonus for exploration

Main Hyperparameters
	•	TOTAL_TIMESTEPS = 1_000_000
	•	ROLLOUT_STEPS = 2048
	•	NUM_EPOCHS = 10
	•	MINIBATCH_SIZE = 256
	•	GAMMA = 0.99
	•	GAE_LAMBDA = 0.95
	•	CLIP_COEF = 0.2
	•	ENT_COEF = 0.01
	•	LR = 3e-4

# How to Run
1. Prepare data:

├── data/

├── Env_develop.py

├── train_ppo_cps.py

├── plot_ppo_results_cps.py

2. Train PPO
Run: train_ppo_cps.py

3. Get the result plot after training
Run: plot_ppo_results_cps.py

# Notes
•	The current scripts use an absolute local Windows path for BASE_DIR. You may want to change this before running.


