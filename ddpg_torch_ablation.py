import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# =========================
# Actor
# =========================
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )

        self.register_buffer(
            "act_low",
            torch.tensor(act_low, dtype=torch.float32),
        )

        self.register_buffer(
            "act_high",
            torch.tensor(act_high, dtype=torch.float32),
        )

    def forward(self, x):
        x = self.model(x)

        return self.act_low + (x + 1.0) * 0.5 * (
            self.act_high - self.act_low
        )


# =========================
# Critic
# =========================
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs, act):
        return self.model(
            torch.cat([obs, act], dim=-1)
        )


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs = np.zeros((size, obs_dim))
        self.next_obs = np.zeros((size, obs_dim))
        self.act = np.zeros((size, act_dim))
        self.rew = np.zeros((size, 1))
        self.done = np.zeros((size, 1))

        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = o2
        self.done[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(
            0,
            self.size,
            size=batch_size,
        )

        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32),
            "act": torch.tensor(self.act[idx], dtype=torch.float32),
            "rew": torch.tensor(self.rew[idx], dtype=torch.float32),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32),
            "done": torch.tensor(self.done[idx], dtype=torch.float32),
        }


# =========================
# Copy PPO Actor -> DDPG Actor
# =========================
def copy_ppo_to_ddpg(ppo_actor, ddpg_actor):
    ppo_layers = [
        m for m in ppo_actor.modules()
        if isinstance(m, nn.Linear)
    ]

    ddpg_layers = [
        m for m in ddpg_actor.modules()
        if isinstance(m, nn.Linear)
    ]

    for src, dst in zip(ppo_layers, ddpg_layers):
        dst.weight.data.copy_(src.weight.data)
        dst.bias.data.copy_(src.bias.data)


# =========================
# Vanilla DDPG Adaptation
# No shared critic
# No warm-up
# No safety guard
# No tolerance restart
# No best-point restart
# No small actor step scaling
# =========================
def ddpg_torch(
    env,
    ppo_actor=None,
    steps=50000,
    batch_size=64,
    update_after=256,
    gamma=0.99,
    tau=0.005,
    actor_lr=3e-3,
    critic_lr=3e-4,
    noise_std=0.05,
):

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low
    act_high = env.action_space.high

    actor = Actor(
        obs_dim,
        act_dim,
        act_low,
        act_high,
    )

    target_actor = Actor(
        obs_dim,
        act_dim,
        act_low,
        act_high,
    )

    if ppo_actor is not None:
        print(">>> Initialize vanilla DDPG actor from PPO")
        copy_ppo_to_ddpg(
            ppo_actor,
            actor,
        )

    target_actor.load_state_dict(
        actor.state_dict()
    )

    print(">>> Vanilla DDPG: using random Q critic")

    critic = Critic(
        obs_dim,
        act_dim,
    )

    target_critic = Critic(
        obs_dim,
        act_dim,
    )

    target_critic.load_state_dict(
        critic.state_dict()
    )

    actor_opt = optim.Adam(
        actor.parameters(),
        lr=actor_lr,
    )

    critic_opt = optim.Adam(
        critic.parameters(),
        lr=critic_lr,
    )

    buffer = ReplayBuffer(
        100000,
        obs_dim,
        act_dim,
    )

    o = env.reset()

    ep_return = 0.0
    episode_returns = []

    print(">>> Vanilla DDPG adaptation starts")
    print(">>> No shared critic")
    print(">>> No warm-up")
    print(">>> No safety guard")
    print(">>> No tolerance restart")
    print(">>> No best-point restart")

    for t in range(steps):

        obs_t = torch.tensor(
            o,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            a = actor(obs_t).squeeze(0).numpy()

        a += noise_std * np.random.randn(act_dim)

        a = np.clip(
            a,
            act_low,
            act_high,
        )

        o2, r, d, _ = env.step(a)

        buffer.store(
            o,
            a,
            r,
            o2,
            d,
        )

        o = o2
        ep_return += r

        if buffer.size > update_after:
            batch = buffer.sample(batch_size)

            with torch.no_grad():
                next_a = target_actor(
                    batch["next_obs"]
                )

                target_q = batch["rew"] + gamma * (
                    1 - batch["done"]
                ) * target_critic(
                    batch["next_obs"],
                    next_a,
                )

            q = critic(
                batch["obs"],
                batch["act"],
            )

            critic_loss = ((q - target_q) ** 2).mean()

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # No warm-up: update actor immediately after update_after.
            actor_loss = -critic(
                batch["obs"],
                actor(batch["obs"]),
            ).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # Standard DDPG target-network soft update.
            for p, tp in zip(
                actor.parameters(),
                target_actor.parameters(),
            ):
                tp.data.copy_(
                    tau * p.data + (1 - tau) * tp.data
                )

            for p, tp in zip(
                critic.parameters(),
                target_critic.parameters(),
            ):
                tp.data.copy_(
                    tau * p.data + (1 - tau) * tp.data
                )

        if d:
            print(
                f"[Vanilla DDPG] Episode done | "
                f"Return: {ep_return:.2f}"
            )

            episode_returns.append(ep_return)

            o = env.reset()
            ep_return = 0.0

    final_actor = actor

    return {
        "best_actor": final_actor,
        "final_actor": final_actor,

        "last_best_actor": final_actor,
        "last_min_exceed_actor": final_actor,

        "critic": critic,
        "target_critic": target_critic,

        "best_reward": max(episode_returns) if len(episode_returns) > 0 else None,
        "last_best_reward": None,
        "last_min_exceedance": None,

        "episode_returns": episode_returns,
        "episode_exceedances": [],

        "restart_count": 0,
        "tolerance_steps": None,
        "safety_return_threshold": None,
    }
