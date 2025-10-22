import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from utils import mlp_actor_critic, placeholder, placeholders, get_vars, count_vars, ReplayBuffer, sample_param

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def ddpg(env, env_idx,
         gamma = 1.0, epochs = 5, pi_lr = 1e-3, q_lr = 1e-3,
         hidden_sizes = (256, 256), activation = tf.nn.relu,
         max_ep_len = 4000, save_freq = 5, steps_per_epoch = 4000,
         replay_size = int(1e8), polyak = 0.995, batch_size = 100,
         update_after = 200, update_every = 200, act_noise = 0.005,
         rand_act_ratio = 0.01):
    
    """
    env: environment
    env_idx: the index of sampled environment    
    gamma: discount factor for future rewards
    epochs: number of epochs
    pi_lr: actor learning rate
    q_lr: critic learning rate
    hidden_sizes: hidden layer size of neural network
    activation: activation function of neural network
    max_ep_len: maximum epoch length
    save_freq: policy saving frequency, every XX epochs
    steps_per_epoch: number of steps in an epoch
    replay_size: Maximum length of replay buffer
    polyak: interpolation factor in polyak averaging for target networks.
    batch_size: batch size for SGD
    start_steps: randomly sample actions before the start_steps
    update_after: start to update the networks after some steps
    update_every: update every XX steps
    act_noise: noise for sampling action
    rand_act_ratio: the percentage of training data on which actions are randomly selected
    """
    
    #random seed
    
    tf.compat.v1.reset_default_graph()
    seed = 0
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

    # randomly sample actions before the start_steps
    start_steps = int(steps_per_epoch * epochs * rand_act_ratio)


    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit_h = env.action_space.high
    act_limit_l = env.action_space.low

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

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    #var_counts = tuple(count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    #print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

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

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(target_init)    
    
    def get_action(o, noise_scale, pi = pi, act_dim = act_dim, act_limit_h = act_limit_h, act_limit_l = act_limit_l):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, act_limit_l, act_limit_h)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
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
                             d_ph: batch['done']
                             }

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
                saver = tf.train.Saver()
                save_path = saver.save(sess, "./model/saved_model_%d.ckpt"%(env_idx))
                print("Model saved in path: %s" % (save_path))
    
    sess.close()

