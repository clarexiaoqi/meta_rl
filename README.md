# RL Environment Updates

continuous_building_environment.py is derived from the original meta_rl/main/continuous_building_environment.py.
	•	Supervisory control: updates Supply Air Temperature (SAT) and Zone Air Temperature (ZAT) setpoints
	•	PI loop: tracks setpoints through airflow modulation, cooling coil, heating coil, and reheat dynamics

test_pi.py is used for PI-loop validation and supervisory control testing.
