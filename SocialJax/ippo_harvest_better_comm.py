import os
import jax
import jax.numpy as jnp
from jax import random
import jax.tree_util as jtu
import optax
from tqdm import trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from socialjax import make

print(f"JAX version: {jax.__version__}")
jax.tree_map = jtu.tree_map
if not hasattr(jax, 'tree'):
    jax.tree = type('tree', (), {'map': jtu.tree_map})()

NUM_AGENTS = 5
TRAINING_EPISODES = 1500
EVAL_EPISODES = 400
MAX_STEPS_PER_EPISODE = 200
ENV_NAME = "harvest_common_open"
SEEDS = [50, 51, 52, 53, 54]
ROLLING_WINDOW = 50

LEARNING_RATE = 3.5e-4
NUM_EPOCHS = 5
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
VF_COEF = 0.5
ENT_COEF = 0.04
HIDDEN_SIZE = 128
MSG_DIM = 4

BROADCAST_TYPE = "majority"
LOG_MESSAGES_EVERY = 50

def relu(x):
    return jnp.maximum(0, x)

def softmax(x):
    x = x - jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x)
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

class SimplePolicyWithComm:
    @staticmethod
    def init_params(rng, obs_dim: int, n_actions: int):
        key1, key2, key3, key4, key5, key6 = random.split(rng, 6)
        params = {
            'w1': random.normal(key1, (obs_dim, HIDDEN_SIZE)) * 0.1,
            'b1': jnp.zeros(HIDDEN_SIZE),
            'w2': random.normal(key2, (HIDDEN_SIZE, HIDDEN_SIZE)) * 0.1,
            'b2': jnp.zeros(HIDDEN_SIZE),
            'w_action': random.normal(key3, (HIDDEN_SIZE, n_actions)) * 0.1,
            'b_action': jnp.zeros(n_actions),
            'w_msg': random.normal(key4, (HIDDEN_SIZE, MSG_DIM)) * 0.1,
            'b_msg': jnp.zeros(MSG_DIM),
            'w_value': random.normal(key5, (HIDDEN_SIZE, 1)) * 0.01,
            'b_value': jnp.zeros(1)
        }
        return params

    @staticmethod
    @jax.jit
    def forward(params, obs):
        x = relu(jnp.dot(obs, params['w1']) + params['b1'])
        x = relu(jnp.dot(x, params['w2']) + params['b2'])
        action_logits = jnp.dot(x, params['w_action']) + params['b_action']
        msg_logits = jnp.dot(x, params['w_msg']) + params['b_msg']
        value = jnp.dot(x, params['w_value']) + params['b_value']
        return action_logits, msg_logits, value

    @staticmethod
    def forward_batch(params, obs_batch):
        x = relu(jnp.dot(obs_batch, params['w1']) + params['b1'])
        x = relu(jnp.dot(x, params['w2']) + params['b2'])
        action_logits = jnp.dot(x, params['w_action']) + params['b_action']
        msg_logits = jnp.dot(x, params['w_msg']) + params['b_msg']
        value = jnp.dot(x, params['w_value']) + params['b_value']
        return action_logits, msg_logits, value

class TrajectoryBuffer:
    def __init__(self):
        self.clear()
    def clear(self):
        self.obs = []
        self.actions = []
        self.messages = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    def add(self, obs, action, message, reward, value, log_prob, done):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.messages.append(int(message))
        self.rewards.append(float(np.asarray(reward)))
        self.values.append(float(np.asarray(value)))
        self.log_probs.append(float(np.asarray(log_prob)))
        self.dones.append(float(np.asarray(done)))
    def get_batch(self):
        obs = np.array(self.obs)
        actions = np.array(self.actions)
        messages = np.array(self.messages)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        log_probs = np.array(self.log_probs)
        dones = np.array(self.dones)
        advantages, gae = [], 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards)-1 else float(values[t+1])
            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = np.array(advantages)
        returns = advantages + values
        return obs, actions, messages, returns, advantages, log_probs

def compute_broadcast(messages, broadcast_type="majority"):
    if broadcast_type == "majority":
        counts = Counter(messages)
        return counts.most_common(1)[0][0]
    else:
        return int(np.mean(messages))

def train_full_cheap_talk(env, rng, n_actions: int, base_obs_dim: int, seed: int):
    print(f"Training full cheap-talk ({TRAINING_EPISODES} episodes, {BROADCAST_TYPE} broadcast)")
    
    obs_dim = base_obs_dim + 1
    trained_params = {}
    optimizers = {}
    opt_states = {}
    for a in range(NUM_AGENTS):
        rng, key = random.split(rng)
        trained_params[a] = SimplePolicyWithComm.init_params(key, obs_dim, n_actions)
        optimizers[a] = optax.adam(LEARNING_RATE)
        opt_states[a] = optimizers[a].init(trained_params[a])
    
    buffers = [TrajectoryBuffer() for _ in range(NUM_AGENTS)]
    all_rewards = [[] for _ in range(NUM_AGENTS)]
    
    message_history = []
    
    log_filename = f"training_log_seed{seed}.csv"
    log_columns = ['episode', 'mean_reward', 'std_reward'] + [f'agent{a}_reward' for a in range(NUM_AGENTS)] + ['rolling_avg_50']
    if os.path.exists(log_filename):
        os.remove(log_filename)
    pd.DataFrame(columns=log_columns).to_csv(log_filename, index=False)
    print(f"Training log: {log_filename}")
    
    pbar = trange(TRAINING_EPISODES, desc="Cheap-talk IPPO")
    for ep in pbar:
        rng, rng_ep = random.split(rng)
        obs, state = env.reset(rng_ep)
        for b in buffers: b.clear()
        ep_rewards = [0.0] * NUM_AGENTS
        current_broadcast = 0
        
        for t in range(MAX_STEPS_PER_EPISODE):
            rng, rng_step = random.split(rng)
            rngs = random.split(rng_step, NUM_AGENTS + 1)
            
            actions = []
            messages = []
            agent_data = []
            
            for a in range(NUM_AGENTS):
                obs_a = np.array(obs[a], dtype=np.float32).flatten()
                obs_aug = np.concatenate([obs_a, [current_broadcast / (MSG_DIM - 1)]])
                
                action_logits, msg_logits, value = SimplePolicyWithComm.forward(trained_params[a], obs_aug)
                action_probs = softmax(action_logits)
                action = int(random.choice(rngs[a], n_actions, p=action_probs))
                action_log_prob = float(jnp.log(action_probs[action]))
                msg_probs = softmax(msg_logits)
                message = int(random.choice(rngs[a], MSG_DIM, p=msg_probs))
                
                actions.append(action)
                messages.append(message)
                val = float(np.asarray(value).item() if hasattr(np.asarray(value), 'item') else np.asarray(value)[0])
                agent_data.append((obs_aug, action, message, val, action_log_prob))
            
            current_broadcast = compute_broadcast(messages, BROADCAST_TYPE)
            
            next_obs, state, reward, done, info = env.step(rng_step, state, actions)
            is_done = done.get("__all__", False) if isinstance(done, dict) else bool(jnp.all(jnp.array(done)))
            
            for a in range(NUM_AGENTS):
                r = float(reward[a]) if isinstance(reward, (list, tuple)) else float(np.asarray(reward)[a])
                ep_rewards[a] += r
                obs_aug, act, msg, val, lp = agent_data[a]
                buffers[a].add(obs_aug, act, msg, r, val, lp, is_done)
            
            obs = next_obs
            if is_done: break
        
        for a in range(NUM_AGENTS):
            all_rewards[a].append(ep_rewards[a])
        
        mean_r = np.mean(ep_rewards)
        std_r = np.std(ep_rewards)
        rolling = np.mean([np.mean(all_rewards[a][-50:]) for a in range(NUM_AGENTS) if len(all_rewards[a]) >= 50]) if len(all_rewards[0]) >= 50 else mean_r
        
        row = {
            'episode': ep,
            'mean_reward': mean_r,
            'std_reward': std_r,
            **{f'agent{a}_reward': ep_rewards[a] for a in range(NUM_AGENTS)},
            'rolling_avg_50': rolling
        }
        pd.DataFrame([row]).to_csv(log_filename, mode='a', header=False, index=False)
        
        window = min(40, len(all_rewards[0]))
        avg = np.mean([np.mean(all_rewards[a][-window:]) for a in range(NUM_AGENTS)])
        pbar.set_postfix({"avg": f"{avg:.1f}"})
        
        if (ep + 1) % LOG_MESSAGES_EVERY == 0:
            msg_counts = Counter(messages)
            message_history.append({
                'episode': ep,
                **{f'msg_{m}': msg_counts.get(m, 0) for m in range(MSG_DIM)}
            })
        
        for a in range(NUM_AGENTS):
            np_obs, acts, msgs, rets, _, old_lps = buffers[a].get_batch()
            obs_b = jnp.array(np_obs, dtype=jnp.float32)
            acts_b = jnp.array(acts)
            rets_b = jnp.array(rets)
            old_lps_b = jnp.array(old_lps)
            
            def loss_fn(params, obs_b, acts_b, rets_b, old_lps_b):
                action_logits, _, val_b = SimplePolicyWithComm.forward_batch(params, obs_b)
                val_b = val_b.squeeze(-1)
                log_p = jnp.log(softmax(action_logits)[jnp.arange(len(acts_b)), acts_b])
                ratio = jnp.exp(log_p - old_lps_b)
                adv = rets_b - val_b
                norm_adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
                pg1 = -norm_adv * ratio
                pg2 = -norm_adv * jnp.clip(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO)
                pol = jnp.mean(jnp.maximum(pg1, pg2))
                val_l = jnp.mean((rets_b - val_b)**2)
                ent = -jnp.mean(jnp.sum(softmax(action_logits) * jnp.log(softmax(action_logits)+1e-8), axis=1))
                return pol + VF_COEF * val_l + ENT_COEF * ent
            
            for _ in range(NUM_EPOCHS):
                grads = jax.grad(loss_fn)(trained_params[a], obs_b, acts_b, rets_b, old_lps_b)
                updates, opt_states[a] = optimizers[a].update(grads, opt_states[a])
                trained_params[a] = optax.apply_updates(trained_params[a], updates)
    
    if message_history:
        msg_df = pd.DataFrame(message_history)
        msg_df.to_csv(f"message_log_seed{seed}.csv", index=False)
        
        plt.figure(figsize=(10, 5))
        for m in range(MSG_DIM):
            plt.plot(msg_df['episode'], msg_df[f'msg_{m}'], label=f'Message {m}', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Count (last step of episode)")
        plt.title(f"Message Distribution Evolution (Seed {seed}) - {BROADCAST_TYPE} broadcast")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"message_evolution_seed{seed}.png", dpi=300, bbox_inches="tight")
        print(f"Saved message evolution plot for seed {seed}")
    
    print(f"Training log saved: {log_filename}")
    return trained_params, base_obs_dim, log_filename

def evaluate_full_cheap_talk(env, trained_params, rng, n_actions, base_obs_dim, use_comm=True):
    label = "full cheap-talk" if use_comm else "no communication"
    print(f"Evaluating {label}...")
    returns = []
    current_rng = rng
    for ep in range(EVAL_EPISODES):
        rng_ep, current_rng = random.split(current_rng)
        obs, state = env.reset(rng_ep)
        ep_return = 0.0
        current_broadcast = 0
        for t in range(MAX_STEPS_PER_EPISODE):
            rng_t, current_rng = random.split(current_rng)
            rngs = random.split(rng_t, NUM_AGENTS + 1)
            actions = []
            for a in range(NUM_AGENTS):
                obs_a = np.array(obs[a], dtype=np.float32).flatten()
                if use_comm:
                    obs_aug = np.concatenate([obs_a, [current_broadcast / (MSG_DIM - 1)]])
                    action_logits, msg_logits, _ = SimplePolicyWithComm.forward(trained_params[a], obs_aug)
                    msg_probs = softmax(msg_logits)
                    message = int(random.choice(rngs[a], MSG_DIM, p=msg_probs))
                    current_broadcast = compute_broadcast([message], BROADCAST_TYPE)
                else:
                    obs_aug = np.concatenate([obs_a, [0.0]])
                    action_logits, _, _ = SimplePolicyWithComm.forward(trained_params[a], obs_aug)
                action_probs = softmax(action_logits)
                action = int(random.choice(rngs[a], n_actions, p=action_probs))
                actions.append(action)
            obs, state, reward, done, info = env.step(current_rng, state, actions)
            ep_return += float(jnp.sum(jnp.array(reward)))
            if (done.get("__all__", False) if isinstance(done, dict) else jnp.all(jnp.array(done))):
                break
        returns.append(ep_return)
        if ep % 80 == 0 and ep > 0:
            print(f"  Episode {ep}/{EVAL_EPISODES} done")
    return np.array(returns)

print(f"Running cheap-talk IPPO on {ENV_NAME} ({BROADCAST_TYPE} broadcast, {len(SEEDS)} seeds)")

all_data = []
for seed_idx, SEED in enumerate(SEEDS):
    print(f"\nSeed {SEED}")
    rng = random.PRNGKey(SEED)
    env = make(ENV_NAME, num_agents=NUM_AGENTS)
    n_actions = env.action_space(0).n
    
    rng, key = random.split(rng)
    obs_sample, _ = env.reset(key)
    base_obs_dim = int(np.prod(obs_sample[0].shape)) if hasattr(obs_sample[0], 'shape') else len(obs_sample[0])
    print(f"Obs dim: {base_obs_dim}, n_actions: {n_actions}")
    
    trained_params, _, _ = train_full_cheap_talk(env, rng, n_actions, base_obs_dim, SEED)
    
    rng, _ = random.split(rng)
    no_comm = evaluate_full_cheap_talk(env, trained_params, rng, n_actions, base_obs_dim, use_comm=False)
    rng, _ = random.split(rng)
    with_comm = evaluate_full_cheap_talk(env, trained_params, rng, n_actions, base_obs_dim, use_comm=True)
    
    print(f"No comm: {no_comm.mean():.1f} ± {no_comm.std():.1f}")
    print(f"Full cheap-talk: {with_comm.mean():.1f} ± {with_comm.std():.1f}")
    
    all_data.append({
        'no_comm': no_comm,
        'with_comm': with_comm
    })
    
    eval_df = pd.DataFrame([{
        'seed': SEED,
        'no_comm_mean': no_comm.mean(),
        'no_comm_std': no_comm.std(),
        'full_cheap_talk_mean': with_comm.mean(),
        'full_cheap_talk_std': with_comm.std(),
        'improvement': with_comm.mean() - no_comm.mean(),
        'improvement_percent': ((with_comm.mean() - no_comm.mean()) / no_comm.mean() * 100) if no_comm.mean() > 0 else 0
    }])
    eval_df.to_csv(f"eval_results_seed{SEED}.csv", index=False)
    print(f"Saved eval results for seed {SEED}")

print("\nFinal results:")
no_means = [d['no_comm'].mean() for d in all_data]
comm_means = [d['with_comm'].mean() for d in all_data]
print(f"No comm: {np.mean(no_means):.1f} ± {np.std(no_means):.1f}")
print(f"Full cheap-talk: {np.mean(comm_means):.1f} ± {np.std(comm_means):.1f}")
print(f"Difference: {np.mean(comm_means) - np.mean(no_means):+.1f}")

plt.figure(figsize=(8, 5))
plt.bar(["No comm", "Full cheap-talk (majority)"],
        [np.mean(no_means), np.mean(comm_means)],
        yerr=[np.std(no_means), np.std(comm_means)],
        capsize=8, color=["#1f77b4", "#ff7f0e"])
plt.ylabel("Mean Episodic Return")
plt.title(f"Harvest + Cheap-Talk ({BROADCAST_TYPE} broadcast)")
plt.grid(axis='y', alpha=0.3)
plt.savefig("harvest_better_cheap_talk_results.png", dpi=300, bbox_inches="tight")
print("Saved harvest_better_cheap_talk_results.png")
print("Done.")