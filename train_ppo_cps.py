import os
import time
import random
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
#                         USER-TUNABLE HYPERPARAMETERS
# =============================================================================

BASE_DIR = r"C:\Users\jbak2\OneDrive - University of Nebraska\Desktop\CPS\Connect_Env_and_basic_RL\Mar_2"
ENV_PY_NAME = "Env_develop"  # Env_develop.py
DATA_FILE = "weather_data_2013_to_2017_summer_pandas.csv"

RUN_NAME = "PPO_CPS_SAT_HeatSP_CoolSP"
SAVE_DIR = "./ppo_runs_cps"

LAST_EPISODE_CSV_NAME = "last_episode_log.csv"
EPISODE_REWARDS_CSV_NAME = "episode_rewards.csv"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_TIMESTEPS = 300_000
ROLLOUT_STEPS = 2048
NUM_EPOCHS = 10
MINIBATCH_SIZE = 256

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
VF_COEF = 0.5
ENT_COEF = 0.0
MAX_GRAD_NORM = 0.5
LR = 3e-4
ANNEAL_LR = True
TARGET_KL = 0.02

HIDDEN_SIZE = 128

# ---- Env parameters ----
DT = 1800.0
START = 0.0
END = 720.0  # 30 days = 720 hours

# You MUST provide these (placeholders here)
C_ENV = 1.0e6
C_AIR = 1.0e5
R_RC = 0.5
R_OE = 2.0
R_ER = 0.7

LB_SET = 22.0
UB_SET = 24.0
MODE_DEADBAND = 0.0

# Reward normalization
E_REF = 2.0
DT_REF = 2.0
ALPHA = 1.0

# Dual-setpoint enforcement
MIN_DEADBAND = 1.0

# Realistic bounds (from RBC IDF)
SAT_LOW, SAT_HIGH = 12.0, 18.0
HEATSP_LOW, HEATSP_HIGH = 15.56, 22.0
COOLSP_LOW, COOLSP_HIGH = 23.89, 29.44


# =============================================================================
#                                  UTILITIES
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
#                                ROLLOUT BUFFER
# =============================================================================
class RolloutBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: str):
        self.size = size
        self.device = device
        self.ptr = 0

        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((size,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((size,), dtype=torch.float32, device=device)

        # NEW: store objective components per step (from env info)
        self.energy_norm = torch.zeros((size,), dtype=torch.float32, device=device)
        self.comfort_norm = torch.zeros((size,), dtype=torch.float32, device=device)

        self.advantages = torch.zeros((size,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((size,), dtype=torch.float32, device=device)

    def reset(self):
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value, energy_norm, comfort_norm):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.energy_norm[i] = energy_norm
        self.comfort_norm[i] = comfort_norm
        self.ptr += 1

    @torch.no_grad()
    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, lam: float):
        gae = 0.0
        for t in reversed(range(self.size)):
            next_nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_minibatches(self, minibatch_size: int):
        idx = torch.randperm(self.size, device=self.device)
        for start in range(0, self.size, minibatch_size):
            mb_idx = idx[start:start + minibatch_size]
            yield (
                self.obs[mb_idx],
                self.actions[mb_idx],
                self.logprobs[mb_idx],
                self.advantages[mb_idx],
                self.returns[mb_idx],
                self.values[mb_idx],
            )


# =============================================================================
#                    ACTOR-CRITIC (tanh-squashed Gaussian)
# =============================================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_low: np.ndarray, act_high: np.ndarray):
        super().__init__()
        self.register_buffer("act_low", torch.tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.tensor(act_high, dtype=torch.float32))

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, act_dim),
        )

        self.log_std = nn.Parameter(torch.ones(act_dim) * (-1.0))  # std ~ 0.37

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

        self.register_buffer("act_scale", (self.act_high - self.act_low) / 2.0)
        self.register_buffer("act_bias", (self.act_high + self.act_low) / 2.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def _squash(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z) * self.act_scale + self.act_bias

    def _logprob_squashed(self, dist: torch.distributions.Normal, z: torch.Tensor) -> torch.Tensor:
        logp_z = dist.log_prob(z).sum(-1)
        eps = 1e-6
        log_det = torch.log(1.0 - torch.tanh(z) ** 2 + eps).sum(-1)
        return logp_z - log_det

    def get_action_and_value(self, obs: torch.Tensor):
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        z = dist.rsample()
        action = self._squash(z)
        logprob = self._logprob_squashed(dist, z)
        value = self.get_value(obs)
        return action, logprob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        y = (actions - self.act_bias) / (self.act_scale + 1e-8)
        y = torch.clamp(y, -0.999999, 0.999999)
        z = 0.5 * torch.log((1 + y) / (1 - y))  # atanh(y)

        logprob = self._logprob_squashed(dist, z)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs)
        return logprob, entropy, value


# =============================================================================
#                           CSV LOGGING HELPERS
# =============================================================================
def flatten_info(info: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in info.items():
        if isinstance(v, (int, float, np.floating, np.integer, bool)):
            out[k] = float(v) if not isinstance(v, bool) else int(v)
        else:
            out[k] = str(v)
    return out


def run_one_episode_and_log(env, model: ActorCritic, device: str) -> pd.DataFrame:
    obs = env.reset()
    done = False
    tstep = 0
    rows: List[Dict[str, Any]] = []

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, _, _ = model.get_action_and_value(obs_t)
        action = action_t.squeeze(0).cpu().numpy()

        next_obs, reward, done, info = env.step(action)

        row = {"tstep": tstep, "reward": float(reward), "done": int(bool(done))}
        for i in range(len(obs)):
            row[f"obs_{i}"] = float(obs[i])
        for j in range(len(action)):
            row[f"action_{j}"] = float(action[j])

        row.update(flatten_info(info))
        rows.append(row)

        obs = next_obs
        tstep += 1

    return pd.DataFrame(rows)


# =============================================================================
#                                TRAINING LOOP
# =============================================================================
def train_ppo(env, model: ActorCritic, optimizer, device: str):
    os.makedirs(SAVE_DIR, exist_ok=True)
    run_dir = os.path.join(SAVE_DIR, f"{RUN_NAME}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    buffer = RolloutBuffer(obs_dim, act_dim, ROLLOUT_STEPS, device)

    obs = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    num_updates = TOTAL_TIMESTEPS // ROLLOUT_STEPS
    global_step = 0

    episode_idx = 0
    ep_return = 0.0
    ep_len = 0

    # NEW: track objective sums per episode (using env info EnergyNorm/ComfortNorm)
    ep_energy_norm_sum = 0.0
    ep_comfort_norm_sum = 0.0

    episode_rows: List[Dict[str, Any]] = []

    for update in range(1, num_updates + 1):
        if ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * LR
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        buffer.reset()

        for _ in range(ROLLOUT_STEPS):
            global_step += 1

            with torch.no_grad():
                action_t, logprob_t, value_t = model.get_action_and_value(obs_t.unsqueeze(0))
                action_t = action_t.squeeze(0)
                logprob_t = logprob_t.squeeze(0)
                value_t = value_t.squeeze(0)

            action = action_t.cpu().numpy()
            next_obs, reward, done, info = env.step(action)

            # pull objective components from env info (must exist in your env)
            en = float(info.get("EnergyNorm", np.nan))
            co = float(info.get("ComfortNorm", np.nan))

            ep_return += float(reward)
            ep_len += 1
            ep_energy_norm_sum += en
            ep_comfort_norm_sum += co

            buffer.add(
                obs=obs_t,
                action=action_t,
                logprob=logprob_t,
                reward=torch.tensor(reward, dtype=torch.float32, device=device),
                done=torch.tensor(float(done), dtype=torch.float32, device=device),
                value=value_t,
                energy_norm=torch.tensor(en, dtype=torch.float32, device=device),
                comfort_norm=torch.tensor(co, dtype=torch.float32, device=device),
            )

            obs = next_obs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            if done:
                # Total reward is -(EnergyNorm + alpha*ComfortNorm), so per-episode components:
                # reward_energy_component = -sum(EnergyNorm)
                # reward_comfort_component = -alpha*sum(ComfortNorm)
                episode_rows.append({
                    "episode": episode_idx,
                    "end_global_step": global_step,
                    "episode_length": ep_len,
                    "update": update,

                    "episode_return_total": ep_return,
                    "episode_return_energy": -ep_energy_norm_sum,
                    "episode_return_comfort": -(ALPHA * ep_comfort_norm_sum),

                    # also store raw sums if you want
                    "sum_EnergyNorm": ep_energy_norm_sum,
                    "sum_ComfortNorm": ep_comfort_norm_sum,
                })

                episode_idx += 1
                ep_return = 0.0
                ep_len = 0
                ep_energy_norm_sum = 0.0
                ep_comfort_norm_sum = 0.0

                obs = env.reset()
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            last_value = model.get_value(obs_t.unsqueeze(0)).squeeze(0)

        buffer.compute_returns_and_advantages(last_value, GAMMA, GAE_LAMBDA)

        adv = buffer.advantages
        buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        approx_kl = 0.0
        clipfracs = []
        pi_losses = []
        v_losses = []
        entropies = []

        for _epoch in range(NUM_EPOCHS):
            for (mb_obs, mb_actions, mb_logprob_old, mb_adv, mb_returns, mb_values_old) in buffer.get_minibatches(
                MINIBATCH_SIZE
            ):
                new_logprob, entropy, new_value = model.evaluate_actions(mb_obs, mb_actions)

                logratio = new_logprob - mb_logprob_old
                ratio = logratio.exp()

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (new_value - mb_returns) ** 2
                v_clipped = mb_values_old + torch.clamp(new_value - mb_values_old, -CLIP_COEF, CLIP_COEF)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_logprob_old - new_logprob).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()

                clipfracs.append(clipfrac)
                pi_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy_loss.item())

            if TARGET_KL is not None and approx_kl > TARGET_KL:
                break

        # logging: use total return
        if len(episode_rows) > 0:
            avg10 = float(np.mean([r["episode_return_total"] for r in episode_rows[-10:]]))
        else:
            avg10 = float("nan")

        print(
            f"[Update {update:04d}/{num_updates}] steps={global_step} "
            f"episodes={episode_idx} avgReturn(10ep)={avg10: .3f} "
            f"piLoss={np.mean(pi_losses): .4f} vLoss={np.mean(v_losses): .4f} "
            f"entropy={np.mean(entropies): .4f} KL={approx_kl: .4f} clipFrac={np.mean(clipfracs): .4f}"
        )

        if update % 10 == 0 or update == num_updates:
            ckpt_path = os.path.join(run_dir, f"ppo_cps_update_{update}.pt")
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "global_step": global_step,
                        "episode_idx": episode_idx}, ckpt_path)

    final_path = os.path.join(run_dir, "ppo_cps_final.pt")
    torch.save({"model_state": model.state_dict()}, final_path)
    print(f"Training done. Final model saved to: {final_path}")

    df_eps = pd.DataFrame(episode_rows)
    eps_csv_path = os.path.join(BASE_DIR, EPISODE_REWARDS_CSV_NAME)
    df_eps.to_csv(eps_csv_path, index=False)
    print(f"Episode rewards saved to: {eps_csv_path}")

    return final_path


# =============================================================================
#                                   MAIN
# =============================================================================
if __name__ == "__main__":
    os.chdir(BASE_DIR)
    set_seed(SEED)

    env_module = __import__(ENV_PY_NAME)
    EnvClass = getattr(env_module, "ContinuousBuildingControlEnvironment")

    env = EnvClass(
        data_file=DATA_FILE,
        dt=DT,
        start=START,
        end=END,
        C_env=C_ENV,
        C_air=C_AIR,
        R_rc=R_RC,
        R_oe=R_OE,
        R_er=R_ER,
        lb_set=LB_SET,
        ub_set=UB_SET,
        mode_deadband=MODE_DEADBAND,
        E_ref=E_REF,
        dT_ref=DT_REF,
        alpha=ALPHA,
        min_deadband=MIN_DEADBAND,
        SAT_low=SAT_LOW,
        SAT_high=SAT_HIGH,
        HeatSP_low=HEATSP_LOW,
        HeatSP_high=HEATSP_HIGH,
        CoolSP_low=COOLSP_LOW,
        CoolSP_high=COOLSP_HIGH,
    )
    env.seed(SEED)

    device = torch.device(DEVICE)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    model = ActorCritic(obs_dim, act_dim, act_low, act_high).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    train_ppo(env, model, optimizer, device=str(device))

    df_last = run_one_episode_and_log(env, model, device=str(device))
    last_csv_path = os.path.join(BASE_DIR, LAST_EPISODE_CSV_NAME)
    df_last.to_csv(last_csv_path, index=False)
    print(f"Last episode log saved to: {last_csv_path}")