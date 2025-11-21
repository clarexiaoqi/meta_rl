from env import ContinuousBuildingControlEnvironment as BEnv
import numpy as np
import os
from utils import sample_param, custom_plot
from ddpg_online import ddpg_online
import tensorflow as tf
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    #set up environment
    data_file ='weather_data_2013_to_2017_summer_pandas.csv'
    policy_file = "./model/saved_model_0.ckpt"
    online_start = 17664.
    online_end = 19872.5
    env = BEnv(data_file, start = online_start, end = online_end,
               C_env = 3.1996e6,
               C_air = 3.5187e5,
               R_rc = 0.00706,
               R_oe = 0.02707,
               R_er = 0.00369,
               lb_set = 22.,
               ub_set = 24.)
    
	#run ddpg
    T_air, time, T_out, Q_SG, action_list, energy_list, penalty_list, \
    temp_metric_list, lb_list, ub_list = \
    ddpg_online(env = env, env_idx = 0, policy_file = policy_file,
                start = online_start, end = online_end,
                gamma = 1.0, epochs = 1, pi_lr = 1e-3, q_lr = 1e-3,
                hidden_sizes = (64, 64, 64, 64), activation = tf.nn.relu,
                max_ep_len = int((online_end - online_start)*2), 
                save_freq = 5, steps_per_epoch = int((online_end - online_start)*2),
                replay_size = int(1e8), polyak = 0.995, batch_size = 50,
                update_after = 100, update_every = 12, act_noise = 0.001,
                rand_act_ratio = 0.)
    
    custom_plot(T_air[2100:2300], time[2100:2300], T_out[2100:2300], Q_SG[2100:2300], 
                action_list[2100:2300], energy_list[2100:2300], penalty_list[2100:2300], 
                lb_list[2100:2300], ub_list[2100:2300], 0, "online")
    
    print("Energy Use in kWh: %d" %energy_list[-1])
    print("# of Hours out of Bounds: %d" %penalty_list[-1])
    print("Temperature Exceedance in degC-hr: %d" %temp_metric_list[-1])
    
    d = {'energy_true': energy_list.T[0,:], 'penalty_true': penalty_list, 'exceedance_true': temp_metric_list}
    df = pd.DataFrame(data = d)
    #df.to_csv('results_true.csv', index = False)

if __name__ == "__main__":
    main()
