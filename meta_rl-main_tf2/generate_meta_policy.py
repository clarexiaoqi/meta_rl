from env import ContinuousBuildingControlEnvironment as BEnv
import numpy as np
import os
from utils import sample_param
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from meta_rl import meta_rl
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    
    env_param = pd.read_csv('env_param.csv')
    num_env = 5
    
    start = 11040.
    end = 17663.
    
    meta_rl(env_param, num_env, start = start, end = end,
            data_file = 'weather_data_2013_to_2017_summer_pandas.csv',
            gamma = 1.0, epochs = 1, pi_lr = 1e-4, q_lr = 1e-4,
            hidden_sizes = (64, 64, 64, 64), activation = tf.nn.relu,
            max_ep_len = int((end - start)*2), save_freq = 5, 
            steps_per_epoch = int((end - start)*2), replay_size = int(1e8), 
            polyak = 0.995, batch_size = 250, update_after = 150, 
            update_every = 48, act_noise = 0.005, start_steps = 500,
            episodes = 5, clip_ratio = 0.2, 
            train_pi_iters = 100, train_v_iters = 100,
            lam = 0.97, target_kl = 0.01)
    
if __name__ == "__main__":
    main()
