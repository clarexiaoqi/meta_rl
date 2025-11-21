from env import ContinuousBuildingControlEnvironment as BEnv
import numpy as np
import os
from utils import sample_param
from ddpg import ddpg
import tensorflow as tf
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    
    #generate environment samples
    #param = sample_param(num_env_sample)
    #print(param)
    #load environment samples
    param = pd.read_csv('env_param2.csv')
    num_env_sample = len(param)
    
    data_file ='weather_data_2013_to_2017_summer_pandas.csv'
    start = 11040.
    end = 17663.
    
    #generate policies from each environment sample
    for sample in range(num_env_sample):
        env = BEnv(data_file, start = start, end = end, 
                   C_env = param['C_env'][num_env_sample - sample - 1],
                   C_air = param['C_air'][num_env_sample - sample - 1],
                   R_rc = param['R_rc'][num_env_sample - sample - 1],
                   R_oe = param['R_oe'][num_env_sample - sample - 1],
                   R_er = param['R_er'][num_env_sample - sample - 1])
        
        
        ddpg(env = env, env_idx = num_env_sample - sample,
             gamma = 1.0, epochs = 10, pi_lr = 1e-3, q_lr = 1e-3,
             hidden_sizes = (64, 64, 64, 64), activation = tf.nn.relu,
             max_ep_len = int((end - start)*2), save_freq = 5, 
             steps_per_epoch = int((end - start)*2),
             replay_size = int(1e8), polyak = 0.995, batch_size = 250,
             update_after = 150, update_every = 48, act_noise = 0.005,
             rand_act_ratio = 0.05)
    
if __name__ == "__main__":
    main()
