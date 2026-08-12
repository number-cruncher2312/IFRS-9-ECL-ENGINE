#!/usr/bin/env python3
"""
validate_pd_pipeline.py
-----------------------
Validate the current borrower-state modification module by running the full
15,000-row cleaned GMSC snapshot through the existing XGBoost PD pipeline.

Steps
1. Load the 15k cleaned GMSC snapshot (origination state)
2. Generate the current snapshot using construct_current_snapshot() with a fixed seed
3. Load the persisted XGBoost model
4. Score both snapshots (origination + current) to get PDs
5. Compare PD distributions and report metrics
6. Persist predictions + summary to outputs/

No generator modifications, no parameter tuning — validation only.
"""

import os
import sys
import json

# Force UTF-8 output on Windows consoles that default to cp1252
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp

# ─── Path setup so we can import project modules regardless of CWD ───────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from synthetic_data.modification import construct_current_snapshot  # noqa: E402

# ─── Constants ───────────────────────────────────────────────────────────────
SNAPSHOT_PATH = os.path.join(
    PROJECT_ROOT, "synthetic_data", "credit_snapshot", "GiveMeSomeCredit-snapshot.csv"
)
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "Credit-Risk-Predictor-main", "model", "xgb_model.pkl"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIXED_SEED = 42
TARGET = "SeriousDlqin2yrs"

# Feature order the model was trained on (must match train_model.py)
FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def load_snapshot() -> pd.DataFrame:
    """Load the 15k cleaned GMSC origination snapshot."""
    if not os.path.exists(SNAPSHOT_PATH):
        raise FileNotFoundError(
            f"Snapshot not found at {SNAPSHOT_PATH}. "
            "Run synthetic_data/extract_snapshot.py first."
        )
    df = pd.read_csv(SNAPSHOT_PATH)
    print(f"Loaded origination snapshot: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def load_model():
    """Load the persisted XGBoost model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run Credit-Risk-Predictor-main/train_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    print(f"Loaded XGBoost model from {MODEL_PATH}")
    return model


def preprocess_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same preprocessing used in train_model.py / app.py:
      - Drop the target column
      - Keep only the feature columns the model was trained on
      - Median-impute any remaining NaNs
    """
    X = df.drop(columns=[TARGET], errors="ignore").copy()

    # Reorder / select feature columns
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = np.nan
    X = X[FEATURE_COLUMNS]

    # Median imputation (matches train_model.py)
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)

    return X


def score_snapshot(model, df: pd.DataFrame) -> np.ndarray:
    """Preprocess + predict_proba → return PD (positive-class probability)."""
    X = preprocess_for_model(df)
    probs = model.predict_proba(X)[:, 1]
    return probs


def psi_score(expected, actual, bins=10):
    """
    Population Stability Index between two distributions.
    """
    # Use quantile bins based on the expected (origination) distribution
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Avoid log(0)
    eps = 1e-6
    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load origination snapshot ───────────────────────────────────────
    orig_df = load_snapshot()

    # ── 2. Generate current snapshot with fixed seed ───────────────────────
    print(f"\nGenerating current snapshot with seed={FIXED_SEED} ...")
    current_df = construct_current_snapshot(orig_df, seed=FIXED_SEED)
    print(f"  Current snapshot: {current_df.shape[0]} rows × {current_df.shape[1]} cols")

    # Save current snapshot
    current_snapshot_path = os.path.join(OUTPUT_DIR, "current_snapshot_15k.csv")
    current_df.to_csv(current_snapshot_path, index=False)
    print(f"  Saved → {current_snapshot_path}")

    # ── 3. Load model ──────────────────────────────────────────────────────
    model = load_model()

    # ── 4. Score both snapshots ────────────────────────────────────────────
    print("\nScoring origination snapshot ...")
    orig_pds = score_snapshot(model, orig_df)
    print(f"  PD range: {orig_pds.min():.4f} – {orig_pds.max():.4f}  (mean={orig_pds.mean():.4f})")

    print("Scoring current snapshot ...")
    curr_pds = score_snapshot(model, current_df)
    print(f"  PD range: {curr_pds.min():.4f} – {curr_pds.max():.4f}  (mean={curr_pds.mean():.4f})")

    # ── 5. Persist predictions ─────────────────────────────────────────────
    orig_out = orig_df.copy()
    orig_out["PD"] = orig_pds
    orig_pred_path = os.path.join(OUTPUT_DIR, "origination_predictions.csv")
    orig_out.to_csv(orig_pred_path, index=False)
    print(f"\nSaved origination predictions → {orig_pred_path}")

    curr_out = current_df.copy()
    curr_out["PD"] = curr_pds
    curr_pred_path = os.path.join(OUTPUT_DIR, "current_predictions.csv")
    curr_out.to_csv(curr_pred_path, index=False)
    print(f"Saved current predictions     → {curr_pred_path}")

    # ── 6. Comparison metrics ──────────────────────────────────────────────
    ks_stat, ks_pval = ks_2samp(orig_pds, curr_pds)
    psi = psi_score(orig_pds, curr_pds)

    # Mean PD shift
    mean_shift = float(curr_pds.mean() - orig_pds.mean())
    median_shift = float(np.median(curr_pds) - np.median(orig_pds))

    # Default-rate proxy using model PD (expected default rate)
    orig_expected_default = float(orig_pds.mean())
    curr_expected_default = float(curr_pds.mean())

    # Trajectory breakdown
    trajectory_summary = {}
    if "trajectory" in current_df.columns:
        for traj in ["stable", "improving", "deteriorating"]:
            mask = current_df["trajectory"] == traj
            if mask.any():
                trajectory_summary[traj] = {
                    "count": int(mask.sum()),
                    "pct": float(mask.mean() * 100),
                    "mean_pd_orig": float(orig_pds[mask.values].mean()),
                    "mean_pd_curr": float(curr_pds[mask.values].mean()),
                }

    summary = {
        "snapshot_rows": int(len(orig_df)),
        "seed": FIXED_SEED,
        "origination": {
            "mean_pd": float(orig_pds.mean()),
            "median_pd": float(np.median(orig_pds)),
            "std_pd": float(orig_pds.std()),
            "min_pd": float(orig_pds.min()),
            "max_pd": float(orig_pds.max()),
            "expected_default_rate": orig_expected_default,
        },
        "current": {
            "mean_pd": float(curr_pds.mean()),
            "median_pd": float(np.median(curr_pds)),
            "std_pd": float(curr_pds.std()),
            "min_pd": float(curr_pds.min()),
            "max_pd": float(curr_pds.max()),
            "expected_default_rate": curr_expected_default,
        },
        "comparison": {
            "mean_pd_shift": mean_shift,
            "median_pd_shift": median_shift,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "psi": psi,
        },
        "trajectories": trajectory_summary,
    }

    summary_path = os.path.join(OUTPUT_DIR, "pd_validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── 7. Console report ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PD PIPELINE VALIDATION RESULTS (15,000-row snapshot)")
    print("=" * 60)
    print(f"  Snapshot rows         : {summary['snapshot_rows']:,}")
    print(f"  Fixed seed            : {summary['seed']}")
    print()
    print("  Origination Snapshot:")
    print(f"    Mean PD             : {summary['origination']['mean_pd']:.4f}")
    print(f"    Median PD           : {summary['origination']['median_pd']:.4f}")
    print(f"    Std PD              : {summary['origination']['std_pd']:.4f}")
    print(f"    Expected default    : {summary['origination']['expected_default_rate']:.4f}")
    print()
    print("  Current Snapshot:")
    print(f"    Mean PD             : {summary['current']['mean_pd']:.4f}")
    print(f"    Median PD           : {summary['current']['median_pd']:.4f}")
    print(f"    Std PD              : {summary['current']['std_pd']:.4f}")
    print(f"    Expected default    : {summary['current']['expected_default_rate']:.4f}")
    print()
    print("  Comparison:")
    print(f"    Mean PD shift       : {summary['comparison']['mean_pd_shift']:+.4f}")
    print(f"    Median PD shift     : {summary['comparison']['median_pd_shift']:+.4f}")
    print(f"    KS statistic        : {summary['comparison']['ks_statistic']:.4f}  (p={summary['comparison']['ks_pvalue']:.2e})")
    print(f"    PSI                 : {summary['comparison']['psi']:.4f}")
    print()

    if trajectory_summary:
        print("  Trajectory Breakdown:")
        for traj, stats in trajectory_summary.items():
            print(f"    {traj:14s}: n={stats['count']:5d} ({stats['pct']:5.1f}%)  "
                  f"PD orig={stats['mean_pd_orig']:.4f} → curr={stats['mean_pd_curr']:.4f}  "
                  f"Δ={stats['mean_pd_curr'] - stats['mean_pd_orig']:+.4f}")

    print()
    print(f"  Summary JSON          → {summary_path}")
    print(f"  Origination preds     → {orig_pred_path}")
    print(f"  Current preds         → {curr_pred_path}")
    print(f"  Current snapshot CSV  → {current_snapshot_path}")
    print("=" * 60)
    print("  Validation complete.")


if __name__ == "__main__":
    main()