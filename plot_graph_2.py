import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

with open("outputs_2/premium_predictions_2021.json") as f:  # or premium_predictions_2021.json
    data = json.load(f)

preds = data["predictions"]
actual    = np.array([p["actual_premium_2021"]    for p in preds])
predicted = np.array([p["predicted_total_premium"] for p in preds])
pct_err   = np.array([p["pct_error"]               for p in preds if p["pct_error"] is not None])
risk      = np.array([p["fire_risk_score"]          for p in preds])
pp        = np.array([p["predicted_pure_premium"]   for p in preds])

fi_names  = list(data["top_features"].keys())
fi_values = list(data["top_features"].values())
fi_names  = [n.replace("_lag1", " (lag1)").replace("_", " ") for n in fi_names]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Task 2 — Insurance Premium Prediction (2021)", fontsize=14, fontweight="bold", y=1.01)

# ── 1. Actual vs Predicted ─────────────────────────────────────────────────
ax = axes[0, 0]
a_m = actual / 1e6
p_m = predicted / 1e6
ax.scatter(a_m, p_m, alpha=0.45, s=18, color="#378ADD", label="ZIP code")
lim = max(a_m.max(), p_m.max()) * 1.05
ax.plot([0, lim], [0, lim], "--", color="#E24B4A", lw=1.5, label="Perfect fit")
ax.set_xlabel("Actual premium ($M)")
ax.set_ylabel("Predicted premium ($M)")
ax.set_title("Actual vs predicted earned premium")
ax.legend(fontsize=9)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)

# ── 2. Error distribution ──────────────────────────────────────────────────
ax = axes[0, 1]
clipped = pct_err[pct_err <= 50]
bins = range(0, 55, 5)
ax.hist(clipped, bins=list(bins), color="#1D9E75", edgecolor="white", linewidth=0.4)
ax.set_xlabel("Absolute % error")
ax.set_ylabel("Number of ZIPs")
ax.set_title("Error distribution")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

# ── 3. Feature importances ─────────────────────────────────────────────────
ax = axes[1, 0]
sorted_pairs = sorted(zip(fi_values, fi_names))
vals, names = zip(*sorted_pairs)
bars = ax.barh(names, [v * 100 for v in vals], color="#534AB7", height=0.6)
ax.set_xlabel("Importance (%)")
ax.set_title("Top 10 feature importances (gradient boosting)")
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f"{v*100:.2f}%", va="center", fontsize=8)

# ── 4. Fire risk vs pure premium ───────────────────────────────────────────
ax = axes[1, 1]
ax.scatter(risk, pp, alpha=0.45, s=18, color="#D85A30")
ax.set_xlabel("Avg fire risk score")
ax.set_ylabel("Predicted pure premium ($/unit)")
ax.set_title("Fire risk score vs predicted pure premium")
ax.set_ylim(0, 10000)

plt.tight_layout()
plt.savefig("results/fig/task2_plots.jpg", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: task2_plots.jpg")