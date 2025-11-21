from env import ContinuousBuildingControlEnvironment as BEnv
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from utils import mlp_actor_critic, mlp_actor_critic_ppo, placeholder, placeholders, get_vars, count_vars, ReplayBuffer, sample_param, PPOBuffer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def meta_rl(env_param, num_env, start = 0., end = 6000.,
            data_file = 'weather_data_2013_to_2017_winter_pandas.csv',
            gamma = 1.0, epochs = 1, pi_lr = 1e-3, q_lr = 1e-3,
            hidden_sizes = (64, 64, 64, 64), activation = tf.nn.relu,
            max_ep_len = 12000, save_freq = 5, steps_per_epoch = 12000,
            replay_size = int(1e8), polyak = 0.995, batch_size = 250,
            update_after = 100, update_every = 96, act_noise = 0.002,
            start_steps = 600, episodes = 5, clip_ratio = 0.2, 
            train_pi_iters = 100, train_v_iters = 100,
            lam = 0.97, target_kl = 0.01):
    
    """
    env_param: environment parameter file, pd.read_csv('env_param.csv')
    num_env: number of sampled environment    
    gamma: discount factor for future rewards
    epochs: number of epochs for DDPG
    pi_lr: actor learning rate
    q_lr: critic learning rate
    hidden_sizes: hidden layer size of neural network
    activation: activation function of neural network
    max_ep_len: maximum epoch length
    save_freq: policy saving frequency, every XX epochs for DDPG
    steps_per_epoch: number of steps in an epoch
    replay_size: Maximum length of replay buffer for DDPG
    polyak: interpolation factor in polyak averaging for target networks for DDPG
    batch_size: batch size for SGD in DDPG
    start_steps: randomly sample actions before the start_steps in DDPG
    update_after: start to update the networks after some steps in DDPG
    update_every: update every XX steps in DDPG
    act_noise: noise for sampling action in DDPG
    start_steps: randomly sample actions before the start_steps
    clipping ratio: for PPO
    train_pi_iters: maximum number of gradient descent steps on policy loss in an epoch for PPO
    train_v_iters: maximum number of gradient descent steps on value function in an epoch for PPO
    lam: advantage estimation discounting factor for PPO
    target_kl: target kl convergence for PPO
    """


    
    """
    Initialize computational graph, set random seed
    """  
    tf.compat.v1.reset_default_graph()
    seed = 0
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    
    
    
    """
    env related info that is universally applicable
    """    
    # Initialize environment
    env = BEnv(data_file, start = start, end = end, 
                   C_env = env_param['C_env'][0],
                   C_air = env_param['C_air'][0],
                   R_rc = env_param['R_rc'][0],
                   R_oe = env_param['R_oe'][0],
                   R_er = env_param['R_er'][0])
    obs_dim = env.observation_space.shape[0]
    act_dim = 1#benv.action_space.shape[0]
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit_h = env.action_space.high[0]
    act_limit_l = env.action_space.low[0]



    """
    Set computational graph for DDPG-learner in each env sample
    """

    obs_dim = env.observation_space.shape[0]
    act_dim = 1#benv.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit_h = env.action_space.high[0]
    act_limit_l = env.action_space.low[0]

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.compat.v1.variable_scope('main'):
        pi, q, q_pi = mlp_actor_critic(x_ph, a_ph, hidden_sizes=hidden_sizes, activation=activation, 
                                       output_activation=tf.nn.tanh, action_space=env.action_space)
    
    # Target networks
    with tf.compat.v1.variable_scope('target'):
    # Note that the action placeholder going to actor_critic here is 
    # irrelevant, because we only need q_targ(s, pi_targ(s)).
        pi_targ, _, q_pi_targ  = mlp_actor_critic(x2_ph, a_ph, hidden_sizes=hidden_sizes, activation=activation, 
                                                  output_activation=tf.nn.tanh, action_space=env.action_space)
        
    # Randomly initialized network for backup at the begining of each env sample
    with tf.compat.v1.variable_scope('backup'):
        pi_backup, q_backup, q_pi_backup = mlp_actor_critic(x_ph, a_ph, hidden_sizes=hidden_sizes, activation=activation, 
                                                            output_activation=tf.nn.tanh, action_space=env.action_space)    


    # Bellman backup for Q function
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)

    # DDPG losses
    pi_loss = -tf.reduce_mean(q_pi)
    q_loss = tf.reduce_mean((q-backup)**2)

    # Separate train ops for pi, q
    pi_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.compat.v1.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.compat.v1.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])
    
    
    # Make backup for q function ready for a new env sample
    backup_q = tf.group([tf.compat.v1.assign(v_backup, v_main)
                             for v_main, v_backup in zip(get_vars('main/q'), get_vars('backup/q_backup'))])
    backup_q_pi = tf.group([tf.compat.v1.assign(v_backup, v_main)
                                for v_main, v_backup in zip(get_vars('main/q_pi'), get_vars('backup/q_pi_backup'))])
    backup_q_op = [backup_q, backup_q_pi]

    
    # Reset q function for a new env sample
    reset_q = tf.group([tf.compat.v1.assign(v_main, v_backup)
                            for v_backup, v_main in zip(get_vars('backup/q_backup'), get_vars('main/q'))])
    reset_q_pi = tf.group([tf.compat.v1.assign(v_main, v_backup)
                               for v_backup, v_main in zip(get_vars('backup/q_pi_backup'), get_vars('main/q_pi'))])
    reset_q_pi_targ = tf.group([tf.compat.v1.assign(v_main, v_backup)
                                    for v_backup, v_main in zip(get_vars('backup/q_pi_backup'), get_vars('target/q_pi_targ'))])
    reset_q_op = [reset_q, reset_q_pi, reset_q_pi_targ]
    
       
    # get action from DDPG
    def get_action(o, noise_scale, pi = pi, act_dim = act_dim, act_limit_h = act_limit_h, act_limit_l = act_limit_l):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, act_limit_l, act_limit_h)


    """
    Set computational graph for PPO-meta-learner
    """
    
    # Inputs to computation graph
    x_ph_meta, a_ph_meta, adv_ph, ret_ph, logp_old_ph = placeholders(obs_dim, act_dim, None, None, None)

    # For collecting data from DDPG learnt policy
    with tf.compat.v1.variable_scope('local'):
        pi_loc, logp_loc, logp_pi_loc, v_loc = mlp_actor_critic_ppo(x_ph_meta, a_ph_meta, hidden_sizes=hidden_sizes, activation=activation, 
                                                output_activation=tf.nn.tanh, action_space=env.action_space)

    # Main outputs from computation graph
    with tf.compat.v1.variable_scope('meta'):
        pi_meta, logp_meta, logp_pi_meta, v_meta = mlp_actor_critic_ppo(x_ph_meta, a_ph_meta, hidden_sizes=hidden_sizes, activation=activation, 
                                                output_activation=tf.nn.tanh, action_space=env.action_space)
    
    # Make backups for ppo networks
    with tf.compat.v1.variable_scope('backup_loc'):
        pi_back, logp_loc_back, logp_pi_loc_back, v_loc_back = mlp_actor_critic_ppo(x_ph_meta, a_ph_meta, hidden_sizes=hidden_sizes, activation=activation, 
                                                output_activation=tf.nn.tanh, action_space=env.action_space)    

    backup_loc_op = tf.group([tf.compat.v1.assign(v_back, v_loc)
                             for v_loc, v_back in zip(get_vars('local'), get_vars('backup_loc'))])

    
    # Reset ppo networks  
    reset_pi_loc = tf.group([tf.compat.v1.assign(v_loc, v_main)
                             for v_main, v_loc in zip(get_vars('main/pi'), get_vars('local/pi_loc'))])
    reset_logp_loc = tf.group([tf.compat.v1.assign(v_loc, v_back)
                              for v_back, v_loc in zip(get_vars('backup_loc/logp_loc_back'), get_vars('local/logp_loc'))])
    reset_logp_pi_loc = tf.group([tf.compat.v1.assign(v_loc, v_back)
                              for v_back, v_loc in zip(get_vars('backup_loc/logp_pi_loc_back'), get_vars('local/logp_pi_loc'))])
    reset_v_loc = tf.group([tf.compat.v1.assign(v_loc, v_back)
                              for v_back, v_loc in zip(get_vars('backup_loc/v_loc_back'), get_vars('local/v_loc'))])
    reset_loc_op = [reset_pi_loc, reset_logp_loc, reset_logp_pi_loc, reset_v_loc]
    
    
    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph_meta, a_ph_meta, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi_loc, v_loc, logp_pi_loc]

    # Experience buffer
    memory = PPOBuffer(obs_dim, act_dim, steps_per_epoch*num_env, gamma, lam)

    # Objective functions
    ratio = tf.exp(logp_meta - logp_old_ph) # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph > 0, (1 + clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
    actor_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    critic_loss = tf.reduce_mean((ret_ph - v_meta)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp_meta) # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp_meta) # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Optimizers
    train_actor = tf.compat.v1.train.AdamOptimizer(learning_rate=pi_lr).minimize(actor_loss)
    train_critic = tf.compat.v1.train.AdamOptimizer(learning_rate=q_lr).minimize(critic_loss)
    
    # Reset actor network for DDPG as meta policy
    reset_pi = tf.group([tf.compat.v1.assign(v_main, v_meta)
                              for v_meta, v_main in zip(get_vars('meta/pi_meta'), get_vars('main/pi'))])
    reset_pi_targ = tf.group([tf.compat.v1.assign(v_targ, v_meta)
                              for v_meta, v_targ in zip(get_vars('meta/pi_meta'), get_vars('target/pi_targ'))])
    reset_pi_op = [reset_pi, reset_pi_targ]
    
    
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(target_init)
    sess.run(backup_q_op)    
    sess.run(backup_loc_op)
    

    """
    meta policy update function with PPO
    """

    def update_meta():
        inputs = {k:v for k,v in zip(all_phs, memory.get())}
        pi_l_old, v_l_old, ent = sess.run([actor_loss, critic_loss, approx_ent], feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_actor, approx_kl], feed_dict=inputs)
            if kl > 1.5 * target_kl:
                break
        for _ in range(train_v_iters):
            sess.run(train_critic, feed_dict=inputs)
        pi_l_new, v_l_new, kl, cf = sess.run([actor_loss, critic_loss, approx_kl, clipfrac], feed_dict=inputs)
        

        
    """
    Loop for iterating over all the env samples
    """ 
    
    for iter in range(episodes):

        for sample in range(num_env):
        
            # Update env parameters 
            env = BEnv(data_file, start = 0., end = 6000., 
                       C_env = env_param['C_env'][sample],
                       C_air = env_param['C_air'][sample],
                       R_rc = env_param['R_rc'][sample],
                       R_oe = env_param['R_oe'][sample],
                       R_er = env_param['R_er'][sample])

            """
            Learn from a sample environment with DDPG
            """
            indicator = -50000.

            max_retry = 3
            best_return = -1e9

            for retry in range(max_retry):
                # Prepare for interaction with a sample env
                total_steps = steps_per_epoch * epochs
                o, ep_ret, ep_len = env.reset(), 0, 0

                # In each sample env, use meta policy to guide learning, and reset q networks
                sess.run([reset_pi_op, reset_q_op])
        
                # Experience buffer for DDPG
                replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

                # Collect experience in a env and update/log each epoch
                for t in range(total_steps):

                    # Until start_steps have elapsed, randomly sample actions
                    # from a uniform distribution for better exploration. Afterwards, 
                    # use the learned policy (with some noise, via act_noise). 
                    if (t > start_steps) & (iter < 1):
                        a = get_action(o, act_noise)
                    else:
                        a = env.action_space.sample()

                    # Step the env
                    o2, r, d, _ = env.step(a)
                    ep_ret += r
                    ep_len += 1

                    # Ignore the "done" signal if it comes from hitting the time
                    # horizon (that is, when it's an artificial terminal signal
                    # that isn't based on the agent's state)
                    d = False if ep_len == max_ep_len else d

                    # Store experience to replay buffer
                    replay_buffer.store(o, a, r, o2, d)

                    # Super critical, easy to overlook step: make sure to update 
                    # most recent observation!
                    o = o2

                    # End of trajectory handling
                    if d or (ep_len == max_ep_len):
                        ep_ret_print = ep_ret
                        o, ep_ret, ep_len = env.reset(), 0, 0

                    # Update handling
                    if t >= update_after and t % update_every == 0:
                        for _ in range(update_every):
                            batch = replay_buffer.sample_batch(batch_size)
                            feed_dict = {x_ph: batch['obs1'],
                                     x2_ph: batch['obs2'],
                                     a_ph: batch['acts'],
                                     r_ph: batch['rews'],
                                     d_ph: batch['done']}

                            # Q-learning update
                            outs = sess.run([q_loss, q, train_q_op], feed_dict)

                            # Policy update
                            outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)

                    # End of epoch wrap-up
                    if (t+1) % steps_per_epoch == 0:
                        epoch = (t+1) // steps_per_epoch
                        print('epoch:', epoch, 'epi_ret:', ep_ret_print)
        
                        # Save model
                        if (epoch % save_freq == 0) or (epoch == epochs):
                            indicator = ep_ret_print
                            saver = tf.compat.v1.train.Saver()
                            save_path = saver.save(sess, "./model/meta_rl/samples/saved_model_%d.ckpt"%(sample))
                            print("Model saved in path: %s" % (save_path))

            """
            Collect trajectory (actions, rewards, observations) based on the learnt DDPG policy to update meta policy
            """
            
            if (np.isnan(ep_ret_print) == False) & (ep_ret_print > -40000):
                
                sess.run(reset_loc_op)

                obs, reward, done, epoch_return, epoch_len = env.reset(), 0, False, 0, 0

                # Main loop: collect experience in env and update/log each epoch
                for t in range(steps_per_epoch):
            
                    action = get_action(obs, act_noise, pi = pi)            
                    _, v_t, logp_t = sess.run(get_action_ops, 
                                   feed_dict={x_ph_meta: obs.reshape(1,-1)})           
                    obs2, reward, done, _ = env.step(action)
                    epoch_return += reward
                    epoch_len += 1
            
                    # save
                    memory.store(obs, action, reward, v_t, logp_t)
            
                    # Update obs (critical!)
                    obs = obs2

                    terminal = done or (epoch_len == max_ep_len)
                    if terminal or (t == steps_per_epoch - 1):                
                        if (sample == num_env - 1):               
                            # if trajectory didn't reach terminal state, bootstrap value target
                            last_val = reward if done else sess.run(v_loc, feed_dict={x_ph_meta: obs.reshape(1,-1)})
                            memory.finish_path(last_val)
                        print('Sample:', sample, 'epi_ret:', epoch_return)
                
                        obs, reward, done, epoch_return, epoch_len = env.reset(), 0, False, 0, 0
                        break

        # Perform PPO update!
        #if (np.isnan(epoch_return) == False) & (epoch_return > -5000): 
        update_meta()
    
        sess.run(reset_pi_op)

        saver = tf.compat.v1.train.Saver()

        save_path = saver.save(sess, "./model/meta_rl/meta/saved_model_%d.ckpt"%(999 + iter))
        print("Model saved in path: %s" % (save_path))
    
    sess.close()
