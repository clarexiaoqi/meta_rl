# test_policy_offline.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.compat.v1.disable_eager_execution()    # Run TF2 in TF1-compatible mode

import numpy as np
import pandas as pd

from env import ContinuousBuildingControlEnvironment as BEnv
from utils import mlp_actor_critic, placeholders, custom_plot


def build_policy_graph(obs_dim, act_dim, env, hidden_sizes=(64, 64, 64, 64)):
    """Build a policy network identical to the training structure (main/pi, main/q)."""
    x_ph, a_ph = placeholders(obs_dim, act_dim)

    with tf.compat.v1.variable_scope("main"):
        pi, q, q_pi = mlp_actor_critic(
            x_ph,
            a_ph,
            hidden_sizes=hidden_sizes,
            activation=tf.nn.relu,
            output_activation=tf.nn.tanh,
            action_space=env.action_space,
        )

    return x_ph, pi


def load_ckpt_partially(sess, ckpt_path):
    """
    Load variables from the checkpoint only if they exist.
    Skip variables that are not found (to avoid mismatch errors such as main/pi/dense_1_1).
    """
    reader = tf.train.load_checkpoint(ckpt_path)
    ckpt_vars = reader.get_variable_to_shape_map()

    loaded = []
    skipped = []

    for var in tf.compat.v1.global_variables():
        name = var.name.split(":")[0]
        if name in ckpt_vars:
            value = reader.get_tensor(name)
            sess.run(var.assign(value))
            loaded.append(name)
        else:
            skipped.append(name)

    print("\n✅ Variables found in checkpoint:")
    for n in sorted(ckpt_vars.keys()):
        print(f"CKPT: {n} {ckpt_vars[n]}")
    for n in skipped:
        if "dense_1_1" in n or "backup" in n:
            print(f"⚠️ Skipped variable not found in ckpt: {n}")

    print(f"✅ Checkpoint successfully loaded from: {ckpt_path}\n")


def test_policy(
    policy_file,
    start=17664.0,
    end=19872.5,
    data_file="weather_data_2013_to_2017_summer_pandas.csv",
):

    # 1) Initialize the environment
    env = BEnv(
        data_file,
        start=start,
        end=end,
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369,
        lb_set=22.0,
        ub_set=24.0,
    )
    obs0, done = env.reset(), False
    obs_dim = env.observation_space.shape[0]
    act_dim = 1
    act_limit_h = env.action_space.high[0]
    act_limit_l = env.action_space.low[0]

    # 2) Build the computation graph
    x_ph, pi = build_policy_graph(obs_dim, act_dim, env)

    # 3) Create session and initialize variables
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # 4) Partially load the pretrained checkpoint
    load_ckpt_partially(sess, policy_file)

    # 5) Rollout and record results
    obs = obs0
    obs_list = []
    reward_list = []
    action_list = []
    u_list = []
    energy_list = [0.0]
    penalty_list = [0.0]
    temp_metric_list = [0.0]
    lb_list = []
    ub_list = []

    while True:
        if not done:
            obs_list.append(obs.copy())
            # Actor generates action
            act = sess.run(pi, feed_dict={x_ph: obs.reshape(1, -1)})[0]
            # Clip action to valid limits
            act = np.clip(act, act_limit_l, act_limit_h)

            obs, r, done, info = env.step(act)

            reward_list.append(r)
            action_list.append(info["a_t"][0])
            u_list.append(info["u_t"][0])
            energy_list.append(info["Energy"][0] + energy_list[-1])
            penalty_list.append(info["Penalty"] + penalty_list[-1])
            temp_metric_list.append(info["Exceedance"] + temp_metric_list[-1])
            lb_list.append(info["lb"])
            ub_list.append(info["ub"])

        if done:
            break

    env.close()
    sess.close()
    tf.compat.v1.reset_default_graph()

    # 6) Denormalize values (same as original repository logic)
    low = np.array([10.0, 18.0, 21.0, -40.0, 0.0, 50.0, 0.0])
    high = np.array([35.0, 27.0, 23.0, 40.0, 1100.0, 180.0, 23.0])

    obs_arr = np.array(obs_list)

    T_air = obs_arr[:, 1] * (high[1] - low[1]) + low[1]
    T_out = obs_arr[:, 3] * (high[3] - low[3]) + low[3]
    Q_SG = obs_arr[:, 4] * (high[4] - low[4]) + low[4]

    # Time axis (0.5-hour intervals)
    time = np.linspace(start, end, len(T_air))

    return (
        T_air,
        time,
        T_out,
        Q_SG,
        np.array(action_list),
        np.array(u_list),
        np.array(energy_list[1:]),
        np.array(penalty_list[1:]),
        np.array(temp_metric_list[1:]),
        lb_list,
        ub_list,
    )


def main():
    data_file = "weather_data_2013_to_2017_summer_pandas.csv"
    ckpt_path = "./model/best/saved_model_summer.ckpt"

    (
        T_air,
        time,
        T_out,
        Q_SG,
        action_list,
        u_list,
        energy_list,
        penalty_list,
        temp_metric_list,
        lb_list,
        ub_list,
    ) = test_policy(
        policy_file=ckpt_path,
        start=17664.0,
        end=19872.5,
        data_file=data_file,
    )

    # Select a short window for visualization (same as original plot)
    s, e = 2100, 2300
    os.makedirs("./plots/offline", exist_ok=True)  # Ensure the directory exists

    custom_plot(
        T_air[s:e],
        time[s:e],
        T_out[s:e],
        Q_SG[s:e],
        action_list[s:e],
        energy_list[s:e],
        penalty_list[s:e],
        lb_list[s:e],
        ub_list[s:e],
        9999,
        "offline",
    )

    # Save evaluation metrics to CSV
    df = pd.DataFrame(
        {
            "energy_true": energy_list,
            "penalty_true": penalty_list,
            "exceedance_true": temp_metric_list,
        }
    )
    df.to_csv("results_true.csv", index=False)

    df_profile = pd.DataFrame(
        {
            "lb": lb_list,
            "ub": ub_list,
            "Qsg": Q_SG,
            "Tout": T_out,
            "Tair_true": T_air,
            "u_true": u_list,
        }
    )
    df_profile.to_csv("results_profile.csv", index=False)

    print(f"Energy Use (kWh): {energy_list[-1]:.2f}")
    print(f"Hours Out of Bounds: {penalty_list[-1]:.2f}")
    print(f"Temperature Exceedance (°C·hr): {temp_metric_list[-1]:.2f}")


if __name__ == "__main__":
    main()








