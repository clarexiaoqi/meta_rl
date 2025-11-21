from env import ContinuousBuildingControlEnvironment as BEnv
import numpy as np
import os
from utils import sample_param
from ddpg import ddpg
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    #set up environment
    data_file ='weather_data_2013_to_2017_summer_pandas.csv'
    start = 11040.
    end = 17663.
    env = BEnv(data_file, start = start, end = end,  
                    C_env = 3.1996e6,
                    C_air = 3.5187e5,
                    R_rc = 0.00706,
                    R_oe = 0.02707,
                    R_er = 0.00369)
    
	#run ddpg
    ddpg(env = env, env_idx = 0,
         gamma = 1.0, epochs = 10, pi_lr = 1e-3, q_lr = 1e-3,
         hidden_sizes = (64, 64, 64, 64), activation = tf.nn.relu,
         max_ep_len = int((end - start)*2), save_freq = 5, 
         steps_per_epoch = int((end - start)*2),
         replay_size = int(1e8), polyak = 0.995, batch_size = 250,
         update_after = 150, update_every = 48, act_noise = 0.005,
         rand_act_ratio = 0.05)

if __name__ == "__main__":
    main()
