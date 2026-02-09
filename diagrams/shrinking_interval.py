import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parameters ---
alpha = 0.25
n = 500
seed = 42

# --- Simulation ---
rng = np.random.default_rng(seed)
j = np.arange(1, n + 2)  # j = 1, ..., n+1
mid = (n + 1) / 2

# P(E_i = 1) decays linearly from 2*alpha to 0 at midpoint, then 0
p = np.maximum(0, 2 * alpha * (1 - (j - 1) / mid))
E = rng.binomial(1, p)
Ebar = np.cumsum(E) / j

# Poorly-ranked: constant error rate 0.35 > alpha
E_poor = rng.binomial(1, 0.35, size=len(j))
Ebar_poor = np.cumsum(E_poor) / j

# Poorly-ranked, adversarial: error rate exactly alpha
E_adv = rng.binomial(1, alpha, size=len(j))
Ebar_adv = np.cumsum(E_adv) / j

# Interval bounds
lower = alpha + (1 - alpha) / j
upper = alpha + (2 - alpha) / j

# --- Plot ---
sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True})
fig, ax = plt.subplots(figsize=(7, 3.5))

ax.fill_between(j, lower, upper, color="steelblue", alpha=0.3, label=r"$\left(\alpha + \frac{1-\alpha}{j},\; \alpha + \frac{2-\alpha}{j}\right]$")
ax.plot(j, upper, color="steelblue", linewidth=1)
ax.plot(j, lower, color="steelblue", linewidth=1, linestyle="--")
ax.plot(j, Ebar, color="black", linewidth=1.2, label=r"$\bar{E}_j$ (well-ranked)")
ax.plot(j, Ebar_poor, color="black", linewidth=1.2, linestyle="--", label=r"$\bar{E}_j$ (poorly-ranked)")
ax.plot(j, Ebar_adv, color="black", linewidth=1.2, linestyle=":", label=r"$\bar{E}_j$ (poorly-ranked, adversarial)")
ax.axhline(alpha, color="gray", linestyle=":", linewidth=0.8, label=r"$\alpha$")

ax.set_xlabel(r"$j$", fontsize=12)
ax.set_ylabel("Error", fontsize=12)
ax.set_ylim(0, 1)
ax.set_xlim(1, n + 1)
ax.set_xticks([1, 100, 200, 300, 400, 500])
ax.legend(frameon=True, fontsize=9, shadow=True)
sns.despine(ax=ax, top=True, right=True)
ax.spines["left"].set_color("black")
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_color("black")
ax.spines["bottom"].set_linewidth(0.8)
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
ax.tick_params(axis="both", which="major", direction="out", length=6, width=0.8, color="black", labelcolor="black")
ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.6, color="black")
fig.tight_layout()
fig.savefig("./shrinking_interval.pdf", bbox_inches="tight")
plt.close()