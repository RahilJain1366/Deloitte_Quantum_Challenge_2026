# 2026 Quantum Sustainability Challenge

## Wildfire Risk and Insurance Premium Modeling

This repository contains the solution for Deloitte's 2026 Quantum Sustainability Challenge, submitted by Rahil Jain (University of Texas at Dallas).

The project addresses two connected objectives:

1. **Task 1A/1B**: Predict wildfire risk at California ZIP-code level using quantum/hybrid machine learning, evaluate performance, and document resource requirements.
2. **Task 2**: Forecast 2021 insurance premiums using historical insurance data (2018–2020) and wildfire risk signals derived from Task 1.

---

## Challenge Context

Wildfire intensity and frequency in California are increasing due to a combination of climate, ecological, and human factors (temperature rise, drought dynamics, hydroclimate volatility, wind events, and land-use effects). This creates growing uncertainty in both hazard modeling and insurance pricing.

This project explores whether quantum-enhanced ML workflows can help capture non-linear risk behavior and support better premium modeling decisions.

---

## Repository Structure

```
.
├── qml_task1_QSVM.ipynb              ← PRIMARY: Hybrid QSVM (headline submission model)
├── qml_task2.py                      ← Task 2 insurance premium ensemble
├── aws/
│   ├── code/
│   │   ├── qml_braket_pennylane_fixed.ipynb   ← BASELINE: VQC barren plateau experiment
│   │   └── cell_output.txt
│   └── results/
│       ├── qml_braket_pennylane_all_cells_ran.ipynb
│       ├── qml_results_v7.json                ← VQC baseline metrics (NOT headline result)
│       ├── best_weights.npz
│       ├── performance_plot.py
│       ├── plot_graph_test.py
│       └── fig/
└── results/
    └── fig/
```

---

## Method Summary

### Task 1: Wildfire Risk Prediction — Two Model Streams

This repository contains **two distinct Task 1 model streams**. They serve different roles in the submission and report different metrics. Do not compare their numbers directly.

---

#### Stream A — Hybrid QSVM (Primary / Headline Model)
**File**: `qml_task1_QSVM.ipynb`

This is the **primary submission model** and the source of all headline metrics reported in the PDF paper.

- Classical SVM pre-filtering selects candidate support vectors.
- Quantum kernel `K(xᵢ, xⱼ) = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|²` computed via 4-qubit ZZFeatureMap circuit.
- Final SVM decision function operates on the precomputed quantum kernel matrix.
- Over 19 million quantum circuit evaluations run on **AWS SV1 State Vector Simulator** via Amazon Braket.

**Headline results (from this stream):**

| Metric | QSVM (Quantum Kernel) |
|---|---|
| ROC-AUC | 0.9476 |
| F1 Score | 0.7507 |
| Wildfire Recall | 75.29% |
| Accuracy | 95.81% |

---

#### Stream B — Hybrid VQC (Baseline / Barren Plateau Experiment)
**File**: `aws/code/qml_braket_pennylane_fixed.ipynb`

This stream implements a Variational Quantum Classifier (VQC) as a **rigorous baseline** whose training failure provides the principled motivation for switching to the QSVM architecture. It is not the primary submission model.

- Local training on PennyLane simulator across 150 epochs and 3 random seeds.
- Weight transfer and prediction/evaluation on **Amazon Braket SV1**.
- All seeds exhibit barren plateau behaviour (persistent loss variance, no convergence).
- Outputs stored in `aws/results/qml_results_v7.json`.

**Baseline results (from this stream — VQC only):**

| Metric | VQC Baseline (v7) |
|---|---|
| Test Accuracy | 0.9133 |
| ROC-AUC | 0.6134 |
| F1 Score | 0.2241 |
| Quantum config | 4 qubits, 3 layers, 38 params |

> **Note**: The lower AUC and F1 for the VQC stream are expected and intentional. These numbers document the barren plateau failure that motivated the QSVM architecture. They are not a sign of a broken pipeline — they are the scientific finding. The QSVM (Stream A) addresses this failure and is the submission's primary result.

---

### Task 2: Insurance Premium Forecasting
**File**: `qml_task2.py`

A two-stage time-series ensemble trained on 2018–2020 insurance data, evaluated on the 2021 holdout set.

- **Stage 1**: Predicts pure premium (Earned Premium / Earned Exposure) using a weighted ensemble:
  - Gradient Boosting (weight 0.50)
  - Random Forest (weight 0.35)
  - Ridge Regression (weight 0.15)
  - QSVM-derived wildfire risk probability appended as an additional feature (from Stream A output)
- **Stage 2**: Reconstructs total premium as `predicted_pure_premium × actual_2021_exposure`

**Results:**

| Metric | Value |
|---|---|
| R² | 0.999 |
| MAPE | < 7% |
| QML Feature Importance | 0.12% (nonzero, additive) |

> **Dependency note**: Task 2 requires the QSVM risk probability output from Stream A (`qml_task1_QSVM.ipynb`) to be generated first. If this file is unavailable, the script falls back to using the dataset's average fire risk score as a substitute — this fallback is classical-only and does not represent the full pipeline. Always run Stream A before Task 2 for the complete quantum-classical integration.

---

## Current Results Summary

| Stream | Model | AUC | F1 | Recall | Role |
|---|---|---|---|---|---|
| Stream A | QSVM (quantum kernel) | 0.9476 | 0.7507 | 75.3% | **Primary result** |
| Stream B | VQC (v7, AWS SV1) | 0.6134 | 0.2241 | — | Baseline / barren plateau evidence |

---

## Data Expectations

The code assumes challenge datasets are available locally (not included in this repository):

- Task 1 notebook expects a `Dataset2.csv` path configured in notebook cells.
- Task 2 script currently points to:
  - `task2_data/Dataset2.csv` (insurance data)
  - QSVM risk output JSON from Stream A

Update paths to your local dataset locations before running.

---

## Environment Setup

Recommended Python version: 3.10+

```bash
pip install numpy pandas scikit-learn matplotlib imbalanced-learn pennylane boto3
```

If running Braket workflows, configure AWS credentials and an S3 bucket for Braket artifacts.

---

## How to Run

### Recommended order

1. **Run Stream A first** (primary QSVM model):
   - Open and run `qml_task1_QSVM.ipynb`
   - This generates the QSVM risk probability output used by Task 2

2. **Optionally run Stream B** (VQC baseline experiment):
   - Open and run `aws/code/qml_braket_pennylane_fixed.ipynb`
   - This reproduces the barren plateau loss curves

3. **Run Task 2**:
   ```bash
   python qml_task2.py
   ```
   Expected outputs:
   - `outputs_2/premium_predictions_2021.csv`
   - `outputs_2/premium_predictions_2021.json`

4. **Generate figures**:
   ```bash
   cd aws/results
   python performance_plot.py
   ```
   ```bash
   python plot_graph_2.py "getting the plots for task 2"
   ```
   

---

## Key Outputs and Figures

| File | Description | Source stream |
|---|---|---|
| `results/fig/wildfire_predictions_2023.png` | QSVM ZIP-code risk map | Stream A |
| `results/fig/qsvm_results_v9.png` | QSVM confusion matrix and metrics | Stream A |
| `results/fig/task2_plots.jpg` | Insurance premium predictions | Task 2 |
| `aws/results/fig/Training_Loss_Over_Epochs.jpg` | VQC barren plateau loss curves | Stream B |
| `aws/results/fig/Distribution_of_Risk_Probabilities.jpg` | Risk probability distribution | Stream B |
| `aws/results/fig/ZipCode_vs_RiskProbability.jpg` | ZIP code vs. risk probability | Stream B |

---

## Submission Mapping to Challenge Requirements

| Requirement | Location |
|---|---|
| Team overview / contact | PDF report, Section I |
| One-page abstract (≤400 words) | PDF report, Section II |
| Algorithm details (Task 1A) | `qml_task1_QSVM.ipynb` + PDF Section III |
| Barren plateau evidence (Task 1B) | `aws/results/qml_results_v7.json` + PDF Section IV-A |
| Classical comparison (Task 1B) | PDF Section V, Table 4 |
| Task 2 results | `qml_task2.py` + PDF Section IV-D |
| Envisioned algorithm | PDF Section VI |

---

## Notes and Limitations

- Dataset files are not versioned in this repo.
- Several file paths in scripts/notebooks are environment-specific and may need local edits.
- AWS SV1 provides exact statevector simulation (no shot noise). Results represent upper bounds on real hardware performance.
- The 4-qubit QSVM architecture accommodates 4 input features only; feature selection via mutual information ranking.
- Task 1 class imbalance (8.4% positive) was addressed via SMOTE oversampling and class-weighted SVM penalty.

---

## License

See LICENSE for usage terms.
