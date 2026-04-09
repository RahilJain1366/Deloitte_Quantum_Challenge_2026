# 2026 Quantum Sustainability Challenge

## Wildfire Risk and Insurance Premium Modeling

This repository contains our solution work for Deloitte's 2026 Quantum Sustainability Challenge.

The project addresses two connected objectives:

1. Task 1A/1B: Predict wildfire risk at California ZIP-code level using quantum/hybrid machine learning, evaluate performance, and document resource requirements.
2. Task 2: Forecast 2021 insurance premiums using historical insurance data (2018-2020) and wildfire risk signals.

## Challenge Context

Wildfire intensity and frequency in California are increasing due to a combination of climate, ecological, and human factors (temperature rise, drought dynamics, hydroclimate volatility, wind events, and land-use effects). This creates growing uncertainty in both hazard modeling and insurance pricing.

This project explores whether quantum-enhanced ML workflows can help capture non-linear risk behavior and support better premium modeling decisions.

## Repository Structure

```
.
├── qml_task1_QSVM.ipynb
├── qml_task2.py
├── aws/
│   ├── code/
│   │   ├── qml_braket_pennylane_fixed.ipynb
│   │   └── cell_output.txt
│   └── results/
│       ├── qml_braket_pennylane_all_cells_ran.ipynb
│       ├── qml_results_v7.json
│       ├── best_weights.npz
│       ├── performance_plot.py
│       ├── plot_graph_test.py
│       └── fig/
└── results/
	 └── fig/
```

## Method Summary

### Task 1: Wildfire Risk Prediction (Quantum ML)

This repo includes two Task 1 streams:

- Hybrid QSVM approach in qml_task1_QSVM.ipynb
  - Classical SVM pre-filtering to find support vectors.
  - Quantum kernel construction on support vectors (ZZ feature map style workflow).
  - QSVM with precomputed quantum kernel.
- Hybrid VQC approach with local training + AWS SV1 inference in aws/code/qml_braket_pennylane_fixed.ipynb
  - Local training on PennyLane simulator.
  - Weight transfer and prediction/evaluation on Amazon Braket SV1.
  - Outputs stored in aws/results/qml_results_v7.json.

### Task 2: Insurance Premium Forecasting

Implemented in qml_task2.py as a two-stage time-series ensemble:

1. Stage 1 predicts pure premium (Earned Premium / Earned Exposure), which is more stable than total premium across reporting shifts.
2. Stage 2 reconstructs total premium as:

	predicted_total_premium = predicted_pure_premium x actual_2021_exposure

Model ensemble in Stage 1:

- Gradient Boosting (weight 0.50)
- Random Forest (weight 0.35)
- Ridge Regression (weight 0.15)

The pipeline also attempts to merge QML wildfire risk outputs into Task 2 features (falling back to average fire risk score if unavailable).

## Current Results (from saved artifacts)

From aws/results/qml_results_v7.json and aws/code/cell_output.txt:

- Task 1 test accuracy: 0.9133
- Task 1 ROC-AUC: 0.6134
- Task 1 F1 score: 0.2241
- Quantum configuration: 4 qubits, 3 layers, 38 trainable parameters
- Decision threshold: 0.5

These values correspond to the v7 hybrid VQC pipeline with SV1 evaluation.

## Data Expectations

The code assumes challenge datasets are available locally (not included in this repository):

- Task 1 notebook expects a Dataset2.csv path configured in notebook cells.
- Task 2 script currently points to:
  - tast2_data/Dataset2.csv (insurance data)
  - optional QML risk JSON files

Update paths to your local dataset locations before running.

## Environment Setup

Recommended Python version: 3.10+

Install core dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib imbalanced-learn pennylane boto3
```

If you run Braket workflows, configure AWS credentials and an S3 bucket for Braket artifacts.

## How to Run

### Task 1 (Notebook workflows)

1. Open and run:
	- qml_task1_QSVM.ipynb (QSVM stream), and/or
	- aws/code/qml_braket_pennylane_fixed.ipynb (v7 hybrid stream).
2. Validate generated outputs in:
	- aws/results/qml_results_v7.json
	- aws/results/best_weights.npz
3. Use plotting scripts to generate visual summaries:

```bash
cd aws/results
python performance_plot.py
```

### Task 2 (Script workflow)

```bash
python qml_task2.py
```

Expected outputs (as coded in qml_task2.py):

- outputs_2/premium_predictions_2021.csv
- outputs_2/premium_predictions_2021.json

## Key Outputs and Figures

Existing generated figures include:

- results/fig/wildfire_predictions_2023.png
- results/fig/qsvm_results_v9.png
- results/fig/task2_plots.jpg
- aws/results/fig/Training_Loss_Over_Epochs.jpg
- aws/results/fig/Distribution_of_Risk_Probabilities.jpg
- aws/results/fig/ZipCode_vs_RiskProbability.jpg

## Submission Mapping to Challenge Requirements

This repository supports the PDF submission requirements as follows:

1. Team overview/contact details: provide in final report front matter.
2. One-page abstract: summarize Task 1 + Task 2 approach and outcomes.
3. Algorithm details and assumptions:
	- Task 1 details in notebooks and saved metrics JSON.
	- Task 2 modeling assumptions and feature engineering in qml_task2.py.
4. Results and clickable code link:
	- include this repository URL and reference figures/JSON outputs.
5. Envisioned algorithm and benefits:
	- discuss scaling path to richer geospatial, weather, and claims feature sets.

## Notes and Limitations

- Dataset files are not versioned in this repo.
- Several file paths in scripts/notebooks are environment-specific and may need local edits.
- Metrics can vary with random seeds, data preprocessing choices, and threshold settings.
- Task 1 class imbalance remains a key challenge; evaluate calibration and threshold tuning in final reporting.

## License

See LICENSE for usage terms.