import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

SEEDS = [50, 51, 52, 53, 54]
DATA_DIR = Path(".")

print(f"Generating plots and stats for seeds {SEEDS}")

train_dfs = []
for seed in SEEDS:
    path = DATA_DIR / f"training_log_seed{seed}.csv"
    if not path.exists():
        print(f"Warning: {path} not found")
        continue
    df = pd.read_csv(path)
    df["seed"] = seed
    train_dfs.append(df)

if not train_dfs:
    raise FileNotFoundError("No training logs found")

train_all = pd.concat(train_dfs, ignore_index=True)

train_stats = train_all.groupby("episode").agg(
    reward_mean=("mean_reward", "mean"),
    reward_std=("mean_reward", "std"),
    rolling_mean=("rolling_avg_50", "mean"),
    rolling_std=("rolling_avg_50", "std")
).reset_index()

first_50 = train_all.sort_values("episode").groupby("seed").head(50)
first_50_per_seed = first_50.groupby("seed")["mean_reward"].mean()
first_50_mean = first_50_per_seed.mean()
first_50_std = first_50_per_seed.std()

last_50 = train_all.sort_values("episode").groupby("seed").tail(50)
last_50_per_seed = last_50.groupby("seed")["mean_reward"].mean()
last_50_mean = last_50_per_seed.mean()
last_50_std = last_50_per_seed.std()

best_rolling_per_seed = train_all.groupby("seed")["rolling_avg_50"].max()
best_rolling_mean = best_rolling_per_seed.mean()
best_rolling_std = best_rolling_per_seed.std()

print("Training stats (mean ± std across seeds):")
print(f"  First 50 episodes: {first_50_mean:.1f} ± {first_50_std:.1f}")
print(f"  Final 50 episodes: {last_50_mean:.1f} ± {last_50_std:.1f}")
print(f"  Best 50-ep rolling avg: {best_rolling_mean:.1f} ± {best_rolling_std:.1f}")

eval_files = sorted(glob.glob(str(DATA_DIR / "eval_results_seed*.csv")))
eval_dfs = []
for f in eval_files:
    try:
        eval_dfs.append(pd.read_csv(f))
    except Exception as e:
        print(f"Could not read {f}: {e}")

if eval_dfs:
    eval_all = pd.concat(eval_dfs, ignore_index=True)

    no_comm_mean = eval_all["no_comm_mean"].mean()
    no_comm_std = eval_all["no_comm_mean"].std()
    comm_mean = eval_all["full_cheap_talk_mean"].mean()
    comm_std = eval_all["full_cheap_talk_mean"].std()
    rel_change = ((comm_mean - no_comm_mean) / no_comm_mean * 100) if no_comm_mean != 0 else 0

    print("Eval results:")
    print(f"  No comm: {no_comm_mean:.1f} ± {no_comm_std:.1f}")
    print(f"  Full cheap talk: {comm_mean:.1f} ± {comm_std:.1f}")
    print(f"  Change: {rel_change:+.1f}%")
else:
    print("No eval files found.")

message_cols = [f"msg_{i}" for i in range(4)]
message_labels = [f"Message {i}" for i in range(4)]

msg_props_list = []
entropy_list = []

for seed in SEEDS:
    path = DATA_DIR / f"message_log_seed{seed}.csv"
    if not path.exists():
        print(f"Warning: {path} not found, skipping messages for seed {seed}")
        continue

    msg_df = pd.read_csv(path)

    counts = msg_df[message_cols]
    total = counts.sum(axis=1).replace(0, np.nan)
    props = counts.div(total, axis=0).fillna(0)
    props["episode"] = msg_df["episode"].values
    msg_props_list.append(props)

    probs = props[message_cols].to_numpy()
    entropy = -np.sum(np.where(probs > 0, probs * np.log2(probs), 0), axis=1)
    entropy_df = pd.DataFrame({
        "episode": msg_df["episode"].values,
        "entropy": entropy,
        "seed": seed
    })
    entropy_list.append(entropy_df)

if msg_props_list:
    msg_props_all = pd.concat(msg_props_list, ignore_index=True)
    msg_stats = msg_props_all.groupby("episode").mean().reset_index()

    entropy_all = pd.concat(entropy_list, ignore_index=True)
    entropy_stats = entropy_all.groupby("episode").agg(
        entropy_mean=("entropy", "mean"),
        entropy_std=("entropy", "std")
    ).reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(train_stats["episode"], train_stats["reward_mean"],
             linewidth=2.5, label="Mean reward", color="#1f77b4")
    plt.fill_between(train_stats["episode"],
                     train_stats["reward_mean"] - train_stats["reward_std"],
                     train_stats["reward_mean"] + train_stats["reward_std"],
                     alpha=0.25, color="#1f77b4", label="±1 std")
    plt.plot(train_stats["episode"], train_stats["rolling_mean"],
             linewidth=2.5, label="Mean rolling avg (50 eps)", color="#ff7f0e")
    plt.fill_between(train_stats["episode"],
                     train_stats["rolling_mean"] - train_stats["rolling_std"],
                     train_stats["rolling_mean"] + train_stats["rolling_std"],
                     alpha=0.25, color="#ff7f0e")
    plt.xlabel("Training Episode")
    plt.ylabel("Per-agent mean reward")
    plt.title(f"Harvest Training Reward (mean ± std across {len(SEEDS)} seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve_multi_seed.png", dpi=300)
    plt.close()
    print("Saved training_curve_multi_seed.png")

    plt.figure(figsize=(12, 6))
    plt.stackplot(
        msg_stats["episode"],
        [msg_stats[col] for col in message_cols],
        labels=message_labels,
        alpha=0.85
    )
    plt.xlabel("Training Episode")
    plt.ylabel("Mean proportion at logged timestep")
    plt.title(f"Message Proportions over Training (mean across {len(SEEDS)} seeds)")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("message_proportions_multi_seed.png", dpi=300)
    plt.close()
    print("Saved message_proportions_multi_seed.png")

    plt.figure(figsize=(12, 5))
    plt.plot(entropy_stats["episode"], entropy_stats["entropy_mean"],
             linewidth=2.5, color="#2ca02c", label="Mean entropy")
    plt.fill_between(entropy_stats["episode"],
                     entropy_stats["entropy_mean"] - entropy_stats["entropy_std"],
                     entropy_stats["entropy_mean"] + entropy_stats["entropy_std"],
                     alpha=0.25, color="#2ca02c", label="±1 std")
    plt.xlabel("Training Episode")
    plt.ylabel("Message entropy (bits)")
    plt.title(f"Broadcast-message Entropy over Training (mean ± std across {len(SEEDS)} seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("message_entropy_multi_seed.png", dpi=300)
    plt.close()
    print("Saved message_entropy_multi_seed.png")

    print("All plots done.")
else:
    print("No message logs, skipping plots.")

print("Finished.")