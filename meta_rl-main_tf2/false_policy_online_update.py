from env import ContinuousBuildingControlEnvironment as BEnv
import numpy as np
import os
from utils import custom_plot
from ddpg_online import ddpg_online
import tensorflow as tf
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    
    #load environment samples
    param = pd.read_csv('env_param.csv')
    num_env_sample = len(param)
       
    #load weather data
    data_file ='weather_data_2013_to_2017_winter_pandas.csv'
    
    online_start = 6000.
    online_end = 8000.
    
    penalty_samples = np.array([])
    energy_samples = np.array([])
    temp_metric_samples = np.array([])
    action_samples = np.array([])    
    #generate policies from each environment sample
    for sample in range(num_env_sample):
        env = BEnv(data_file, start = online_start, end = online_end,
                   C_env = 3.1996e6,
                   C_air = 3.5187e5,
                   R_rc = 0.00706,
                   R_oe = 0.02707,
                   R_er = 0.00369)
        
        # load policy file
        policy_file = "./model/saved_model_"+str(sample+1)+".ckpt"
        
        
        #run ddpg and get outputs
        T_air, time, T_out, Q_SG, action_list, energy_list, penalty_list, \
        temp_metric_list, lb_list, ub_list = \
        ddpg_online(env = env, env_idx = sample + 1, policy_file = policy_file,
                    start = online_start, end = online_end,
                    gamma = 1.0, epochs = 1, pi_lr = 1e-3, q_lr = 1e-3,
                    hidden_sizes = (64, 64, 64, 64), activation = tf.nn.relu,
                    max_ep_len = 4000, save_freq = 5, steps_per_epoch = 4000,
                    replay_size = int(1e8), polyak = 0.995, batch_size = 100,
                    update_after = 12, update_every = 12, act_noise = 0.001,
                    rand_act_ratio = 0.)
        
        #plot typical days
        custom_plot(T_air[2100:2300], time[2100:2300], T_out[2100:2300], Q_SG[2100:2300], 
                    action_list[2100:2300], energy_list[2100:2300], penalty_list[2100:2300], 
                    lb_list[2100:2300], ub_list[2100:2300], sample + 1, "online")
        
        
        penalty_samples = np.append(penalty_samples, sum(penalty_list))
        energy_samples = np.append(energy_samples, sum(energy_list))
        temp_metric_samples = np.append(temp_metric_samples, sum(temp_metric_list))
        action_samples = np.append(action_samples, sum(action_list)/1000)  
        
        #show outputs
        print(sample)
        print("Energy Use in kWh: %d" %sum(energy_list))
        print("# of Hours out of Bounds: %d" %sum(penalty_list))
        print("Temperature Exceedance in degC-hr: %d" %sum(temp_metric_list))
    
    # Save data for analysis
    param['Action_total_online'] = action_samples
    param['Energy_online'] = energy_samples
    param['Penalty_online'] = penalty_samples
    param['Exceedance_online'] = temp_metric_samples
    param.to_csv('env_param.csv', index = False)
    #return penalty_samples, energy_samples, temp_metric_samples, action_samples
if __name__ == "__main__":
    #penalty_samples, energy_samples, temp_metric_samples, action_samples = main()
    main()
