# Credit Risk Predictor

A credit risk modeling project using **XGBoost** to predict the probability of a borrower experiencing financial distress.

---

## Model Performance
Based on the *Kaggle "Give Me Some Credit"* dataset, the model achieves strong predictive power:

- **AUC-ROC**: `0.8653`
- **Gini Coefficient**: `0.7306`
- **KS Statistic**: `0.5766`

---

## Tech Stack
- **Modeling**: XGBoost, Scikit-Learn
- **Data Engineering**: Pandas, NumPy
- **Persistence**: Joblib

## Repository Structure
- `train_model.py`: End-to-end training pipeline (Cleaning -> Imputation -> Oversampling -> Training -> Evaluation).
- `model/`: Serialized XGBoost model (`xgb_model.pkl`).
- `data/`: Dataset location and sourcing notes.
- `requirements.txt`: Python dependencies for the model.

## Dataset
This repository does **not** version large raw datasets.

1. Download the *Give Me Some Credit* training file (`cs-training.csv`) from Kaggle:
   https://www.kaggle.com/c/GiveMeSomeCredit/data
2. Place it in either:
   - `data/cs-training.csv` (preferred)
   - `cs-training.csv` (backward-compatible fallback)

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/number-cruncher2312/Credit-Risk-Predictor.git
   cd Credit-Risk-Predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and place the dataset (see **Dataset** section).
4. Train the model:
   ```bash
   python train_model.py
   ```
   This produces `model/xgb_model.pkl`.

5. Load and use the model in Python:
   ```python
   import joblib
   import pandas as pd

   model = joblib.load("model/xgb_model.pkl")
   # Prepare input DataFrame with the expected feature columns
   prediction = model.predict_proba(input_df)[:, 1]
   ```

---
*Created with Love by Antigravity.*