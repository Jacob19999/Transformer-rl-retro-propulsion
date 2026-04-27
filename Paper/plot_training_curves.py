"""
Generate a single clean training curve for the TVC RL paper.

Shows mean episode reward vs. environment steps — the clearest signal
that PPO is learning altitude control and soft landing.

Output: Paper/figures/training_curves.pdf  (and .png)
"""

import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
script_dir = pathlib.Path(__file__).resolve().parent
repo_root = script_dir.parent
run_dir = repo_root / "simulation" / "isaac" / "runs" / "ppo_landing_seed0_20260426_192920"

# ── Load train log ───────────────────────────────────────────────────────────
records = []
with open(run_dir / "train_log.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

steps = np.array([r["global_step"] for r in records]) / 1e6
reward_mean = np.array([r["reward_mean"] for r in records])

# ── Smooth (exponential moving average for a clean curve) ────────────────────
alpha = 0.08
ema = np.zeros_like(reward_mean)
ema[0] = reward_mean[0]
for i in range(1, len(reward_mean)):
    ema[i] = alpha * reward_mean[i] + (1 - alpha) * ema[i - 1]

# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

fig, ax = plt.subplots(figsize=(3.5, 2.4))

c_blue = "#1f77b4"
c_gray = "#999999"

ax.plot(steps, reward_mean, color=c_blue, linewidth=0.35, alpha=0.25)
ax.plot(steps, ema, color=c_blue, linewidth=1.5, label="Mean episode reward (EMA)")
ax.axhline(0, color=c_gray, linestyle=":", linewidth=0.6)

ax.set_xlabel("Environment Steps (millions)")
ax.set_ylabel("Mean Episode Reward")
ax.set_xlim(0, steps[-1])
ax.legend(loc="lower right", framealpha=0.9, edgecolor="none", fontsize=8)
ax.grid(True, alpha=0.25, linewidth=0.5)

fig.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = script_dir / "figures"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "training_curves.pdf")
fig.savefig(out_dir / "training_curves.png")
print(f"Saved to {out_dir / 'training_curves.pdf'} and .png")
plt.close(fig)
