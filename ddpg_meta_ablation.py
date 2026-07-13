import numpy as np
import torch
import torch.optim as optim

from ddpg_torch import Actor, Critic, ReplayBuffer


# =========================
# Copy PPO Actor -> DDPG Actor
# =========================
def copy_ppo_to_ddpg(ppo_actor, ddpg_actor):
    ppo_layers = [
        m for m in ppo_actor.modules()
        if isinstance(m, torch.nn.Linear)
    ]

    ddpg_layers = [
        m for m in ddpg_actor.modules()
        if isinstance(m, torch.nn.Linear)
    ]

    for src, dst in zip(ppo_layers, ddpg_layers):
        dst.weight.data.copy_(src.weight.data)
        dst.bias.data.copy_(src.bias.data)


# =========================
# Evaluate Actor
# =========================
def evaluate_actor(env, actor, device="cpu", steps=720):
    obs = env.reset()
    total_reward = 0.0

    for _ in range(steps):
        obs_t = torch.tensor(
            obs,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            action = actor(obs_t).squeeze(0).cpu().numpy()

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward


# =========================
# Vanilla DDPG Adaptation
# No warm-up, no shared critic, no shared critic optimizer.
# =========================
def ddpg_adapt(
    env,
    ppo_actor=None,
    device="cpu",
    steps=10000,
    batch_size=64,
    gamma=0.99,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-4,
    noise_std=0.1,
):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    actor = Actor(
        obs_dim,
        act_dim,
        act_low,
        act_high,
    ).to(device)

    target_actor = Actor(
        obs_dim,
        act_dim,
        act_low,
        act_high,
    ).to(device)

    if ppo_actor is not None:
        copy_ppo_to_ddpg(
            ppo_actor,
            actor,
        )

    target_actor.load_state_dict(
        actor.state_dict()
    )

    print(">> Using random Q critic")

    critic = Critic(
        obs_dim,
        act_dim,
    ).to(device)

    target_critic = Critic(
        obs_dim,
        act_dim,
    ).to(device)

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

    obs = env.reset()

    for t in range(steps):
        obs_t = torch.tensor(
            obs,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            action = actor(obs_t).squeeze(0).cpu().numpy()

        action += noise_std * np.random.randn(act_dim)
        action = np.clip(
            action,
            act_low,
            act_high,
        )

        next_obs, reward, done, _ = env.step(action)

        buffer.store(
            obs,
            action,
            reward,
            next_obs,
            done,
        )

        obs = next_obs

        if buffer.size > batch_size:
            batch = buffer.sample(batch_size)

            o = batch["obs"].to(device)
            a = batch["act"].to(device)
            r = batch["rew"].to(device)
            o2 = batch["next_obs"].to(device)
            d = batch["done"].to(device)

            with torch.no_grad():
                a2 = target_actor(o2)
                q_target = r + gamma * (1.0 - d) * target_critic(o2, a2)

            q = critic(o, a)
            loss_q = ((q - q_target) ** 2).mean()

            critic_opt.zero_grad()
            loss_q.backward()
            critic_opt.step()

            # No warm-up: actor is updated as soon as the replay buffer is ready.
            loss_pi = -critic(o, actor(o)).mean()

            actor_opt.zero_grad()
            loss_pi.backward()
            actor_opt.step()

            for p, tp in zip(actor.parameters(), target_actor.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

        if done:
            obs = env.reset()

    final_return = evaluate_actor(
        env,
        actor,
        device,
    )

    return actor, final_return
