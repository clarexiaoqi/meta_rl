from env import ContinuousBuildingControlEnvironment as BEnvTrain
from env import ContinuousBuildingControlEnvironment as BEnvTest
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from utils import sample_param

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data_file = 'weather_data_2013_to_2017_winter_pandas.csv'

# 1) Generate parameter samples
num_env_sample = 100
param = sample_param(num_env_sample)
print(param)

df = pd.DataFrame({
    'C_env': param[:, 0],
    'C_air': param[:, 1],
    'R_rc': param[:, 2],
    'R_oe': param[:, 3],
    'R_er': param[:, 4]
})
df.to_csv('env_param.csv', index=False)
param = pd.read_csv('env_param.csv')

# 2) Normalization bounds for the observation space (consistent with main code)
low = np.array([10.0, 18.0, 21.0, -40.0, 0.,   50.,  0. ])
high = np.array([35.0, 27.0, 23.0,  40.0, 1100., 180., 23.])

# 3) Test environment quality using a winter dataset segment
start = 0.0
end = 2000.0

# Random actions: 1 action every 0.5 hr → multiply by 2
T = int((end - start) * 2)
a = np.random.uniform(low=0.0, high=1.0, size=T)

# 4) Select one "true parameter set" and build the true environment
base = param.iloc[0]  # First row of parameters
env_true = BEnvTest(
    data_file,
    start=start,
    end=end,
    C_env=base['C_env'],
    C_air=base['C_air'],
    R_rc=base['R_rc'],
    R_oe=base['R_oe'],
    R_er=base['R_er'],
)

# Roll out the true environment and obtain the “true” T_air sequence
T_air = np.array([])
obs = env_true.reset()
for i in range(T):
    obs, r, done, info = env_true.step(a[i])  # Only unpack 4 return values
    s_t = obs * (high - low) + low           # De-normalize
    T_air = np.append(T_air, s_t[1])

    if done:
        break

# 5) For each parameter sample, build a false environment and compute RMSE
RMSE = np.array([])

for sample in range(len(param)):
    env_false = BEnvTrain(
        data_file,
        start=start,
        end=end,
        C_env=param.loc[sample, 'C_env'],
        C_air=param.loc[sample, 'C_air'],
        R_rc=param.loc[sample, 'R_rc'],
        R_oe=param.loc[sample, 'R_oe'],
        R_er=param.loc[sample, 'R_er'],
    )

    T_air_false = np.array([])
    obs = env_false.reset()
    for t in range(T):
        obs, r, done, info = env_false.step(a[t])
        s_t = obs * (high - low) + low
        T_air_false = np.append(T_air_false, s_t[1])

        if done:
            break

    # Ensure consistent length (safety check)
    min_len = min(len(T_air_false), len(T_air))
    RMSE_sample = np.mean(
        np.sqrt((T_air_false[:min_len] - T_air[:min_len]) ** 2.0)
    )
    RMSE = np.append(RMSE, RMSE_sample)

    print(f"sample {sample}: RMSE = {RMSE_sample:.4f}")

# 6) Save back to CSV
param['RMSE'] = RMSE
param.to_csv('env_param.csv', index=False)
print("Saved env_param.csv with RMSE column.")
