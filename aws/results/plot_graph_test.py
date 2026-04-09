import json
import matplotlib.pyplot as plt

with open("qml_results_v7.json", "r") as f:
    data = json.load(f)

# ── Loss Curve (all 3 seeds) ──────────────────────────────────────────────
loss_histories = data["loss_histories"]

plt.figure()
for i, hist in enumerate(loss_histories):
    plt.plot(range(1, len(hist) + 1), hist, label=f"Seed {i+1}")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("fig/Training_Loss_Over_Epochs.jpg")
plt.show()

# ── Risk Probability Distribution ────────────────────────────────────────
risk_probs = [entry["risk_probability"] for entry in data["predictions_2023"]]
zip_codes  = [entry["zip"]              for entry in data["predictions_2023"]]

plt.figure()
plt.hist(risk_probs, bins=20)
plt.title("Distribution of Risk Probabilities")
plt.xlabel("Risk Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("fig/Distribution_of_Risk_Probabilities.jpg")
plt.show()

# ── Zip Code vs Risk Probability ─────────────────────────────────────────
plt.figure()
plt.scatter(zip_codes, risk_probs, s=5, alpha=0.5)
plt.title("Zip Code vs Risk Probability")
plt.xlabel("Zip Code")
plt.ylabel("Risk Probability")
plt.grid(True)
plt.savefig("fig/ZipCode_vs_RiskProbability.jpg")
plt.show()