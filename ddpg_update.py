import argparse
import random
import torch
import numpy as np
import os

from Env_develop import ContinuousBuildingControlEnvironment as BEnv
from ddpg_torch import ddpg_torch, Critic
from train_ppo_cps import ActorCritic


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed, default is 42.",
    )

    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable plt.show() for batch experiments.",
    )

    args = parser.parse_args()

    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 60)
    print(f"Running DDPG adaptation with seed = {seed}")
    print("=" * 60)

    # =========================
    # 1. Create new environment

    # =========================
    # 1. Create new environment
    # =========================
    data_file = "weather_data_2013_to_2017_summer_pandas.csv"

    env = BEnv(
        data_file=data_file,
        dt=1800.0,
        start=17664,
        end=19872.5,
        # end=19872.5
        C_env=3.1996e6,
        C_air=3.5187e5,
        R_rc=0.00706,
        R_oe=0.02707,
        R_er=0.00369,
    )

    env.seed(seed)
    print(">>> New environment ready")

    # =========================
    # 2. Get dimensions
    # =========================
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    # =========================
    # 3. Load trained PPO general actor
    # =========================
    model = ActorCritic(
        obs_dim,
        act_dim,
        act_low,
        act_high,
    )

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model",
        "final_actor.pth"
    )

    actor_state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    model.actor.load_state_dict(actor_state)
    model.eval()

    print(">>> Loaded general PPO actor")

    # =========================
    # 4. Load trained shared DDPG critic
    # =========================
    critic_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model",
        "shared_ddpg_critic.pth"
    )

    shared_critic = Critic(
        obs_dim,
        act_dim,
    )

    shared_target_critic = Critic(
        obs_dim,
        act_dim,
    )

    critic_checkpoint = torch.load(
        critic_path,
        map_location="cpu",
        weights_only=False,
    )

    shared_critic.load_state_dict(
        critic_checkpoint["shared_critic"]
    )

    shared_target_critic.load_state_dict(
        critic_checkpoint["shared_target_critic"]
    )

    print(">>> Loaded shared DDPG critic")

    # =========================
    # 5. DDPG adaptation setting
    # =========================
    total_steps = 5000000
    warmup_steps = 7*24

    print(">>> DDPG adaptation starts")
    print(">>> Using pretrained/shared Q")
    print(f">>> Warm-up steps: {warmup_steps}")

    # =========================
    # 6. DDPG adaptation
    # =========================
    result = ddpg_torch(
        env,
        ppo_actor=model.actor,
        steps=total_steps,
        warmup_steps=warmup_steps,
        pretrained_critic=shared_critic,
        pretrained_target_critic=shared_target_critic,
    )

    best_actor = result["best_actor"]
    final_actor = result["final_actor"]
    best_reward = result["best_reward"]
    returns = result["episode_returns"]

    last_best_actor = result["last_best_actor"]
    last_min_exceed_actor = result["last_min_exceed_actor"]

    last_best_reward = result["last_best_reward"]
    last_min_exceedance = result["last_min_exceedance"]

    adapted_critic = result["critic"]
    adapted_target_critic = result["target_critic"]

    # =========================
    # 7. Print results
    # =========================
    print("\n=========================")
    print(">>> Best Adaptation Reward:", best_reward)

    if len(returns) > 0:
        print(">>> Final Episode Reward:", returns[-1])
    else:
        print(">>> No episode finished")

    print(">>> Last Best Reward:", last_best_reward)
    print(">>> Last Min Exceedance:", last_min_exceedance)
    print("=========================\n")

    # =========================
    # 8. Save adapted models
    # =========================
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ddpg"
    )
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        "ddpg_adapted.pt",
    )

    torch.save(
        {
            "best_actor": best_actor.state_dict(),
            "final_actor": final_actor.state_dict(),

            "last_best_actor": last_best_actor.state_dict(),
            "last_min_exceed_actor": last_min_exceed_actor.state_dict(),

            "critic": adapted_critic.state_dict(),
            "target_critic": adapted_target_critic.state_dict(),

            "best_reward": best_reward,
            "last_best_reward": last_best_reward,
            "last_min_exceedance": last_min_exceedance,

            "returns": returns,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "seed": seed,

            "actor_init": "final_actor.pth",
            "critic_init": "shared_ddpg_critic.pth",
        },
        save_path,
    )

    print(f">>> Saved adapted models to {save_path}")

    # =========================
    # 9. Print full return curve
    # =========================
    print("\n>>> Episode Returns:")

    for i, r in enumerate(returns):
        print(f"Episode {i + 1}: {r:.2f}")

    # =========================
    # 10. Plot return curve
    # =========================
    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(returns) + 1)
    best_returns_so_far = np.maximum.accumulate(returns)

    plt.figure(figsize=(6.4, 4.0))

    # Best return so far
    plt.plot(
        episodes,
        best_returns_so_far,
        linestyle="-",
        linewidth=1.5,
        marker="s",
        markersize=4,
        label="Best Return So Far",
    )

    # Current return
    plt.plot(
        episodes,
        returns,
        linestyle="--",
        linewidth=1.5,
        marker="o",
        markersize=4,
        label="Current Return",
    )

    plt.xlabel(
        "Adaptation Iteration",
        fontsize=14,
    )

    plt.ylabel(
        "Episode Return",
        fontsize=14,
    )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(
        fontsize=10,
        loc="lower right",
    )

    plt.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.25,
    )

    plt.margins(x=0.03)

    # Final selected policy
    best_idx = len(episodes) - 1

    plt.tight_layout()

    plot_png_path = os.path.join(
        save_dir,
        "ddpg_return_curve.png",
    )

    plot_pdf_path = os.path.join(
        save_dir,
        "ddpg_return_curve.pdf",
    )

    plt.savefig(
        plot_png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        plot_pdf_path,
        bbox_inches="tight",
    )

    if args.no_plot:
        plt.close()
    else:
        plt.show()

    print(f">>> Saved return curve to {plot_png_path}")
    print(f">>> Saved return curve to {plot_pdf_path}")


if __name__ == "__main__":
    main()