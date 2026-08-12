#!/usr/bin/env python3
"""
analyze_stable_trajectory.py
----------------------------
Focused diagnostic on the *stable* trajectory borrowers from the PD validation.

1. Compute PD_change percentiles for stable borrowers only.
2. Identify ~10 stable borrowers with the largest positive PD_change.
3. Show before/after borrower variables alongside PD_origin and PD_current.
4. Determine whether large stable PD increases are driven by delinquency
   changes or other variable movements.
5. Report whether the stable trajectory appears reasonable or systematically
   too risk-increasing.

No generator modifications, no parameter tuning — analysis only.
"""

import os
import sys

# Force UTF-8 output on Windows consoles that default to cp1252
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ─── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

ORIG_PRED_PATH = os.path.join(OUTPUT_DIR, "origination_predictions.csv")
CURR_PRED_PATH = os.path.join(OUTPUT_DIR, "current_predictions.csv")

# Borrower variables to compare before/after
COMPARE_VARS = [
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

# Delinquency-related variables
DELINQUENCY_VARS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


def main():
    # ── Load predictions ───────────────────────────────────────────────────
    orig_df = pd.read_csv(ORIG_PRED_PATH)
    curr_df = pd.read_csv(CURR_PRED_PATH)

    print(f"Loaded origination predictions: {len(orig_df)} rows")
    print(f"Loaded current predictions:     {len(curr_df)} rows")

    # ── Filter to stable trajectory ────────────────────────────────────────
    if "trajectory" not in curr_df.columns:
        raise ValueError("'trajectory' column not found in current predictions.")

    stable_mask = curr_df["trajectory"] == "stable"
    stable_orig = orig_df[stable_mask.values].copy()
    stable_curr = curr_df[stable_mask].copy()

    # Reset index for alignment
    stable_orig = stable_orig.reset_index(drop=True)
    stable_curr = stable_curr.reset_index(drop=True)

    n_stable = len(stable_orig)
    print(f"\nStable trajectory borrowers: {n_stable} ({n_stable/len(orig_df)*100:.1f}%)")

    # ── 1. PD_change percentiles ───────────────────────────────────────────
    pd_change = stable_curr["PD"].values - stable_orig["PD"].values

    percentiles = [0, 10, 25, 50, 75, 90, 100]
    pct_values = np.percentile(pd_change, percentiles)

    print("\n" + "=" * 60)
    print("  1. PD_change PERCENTILES (stable trajectory only)")
    print("=" * 60)
    labels = ["min", "p10", "p25", "median", "p75", "p90", "max"]
    for label, val in zip(labels, pct_values):
        print(f"    {label:8s}: {val:+.4f}")
    print(f"\n    Mean PD_change : {pd_change.mean():+.4f}")
    print(f"    Std  PD_change : {pd_change.std():.4f}")

    # Additional context: how many stable borrowers have positive vs negative change
    pos = (pd_change > 0).sum()
    neg = (pd_change < 0).sum()
    zero = (pd_change == 0).sum()
    print(f"\n    Positive PD_change: {pos} ({pos/n_stable*100:.1f}%)")
    print(f"    Negative PD_change: {neg} ({neg/n_stable*100:.1f}%)")
    print(f"    Zero PD_change    : {zero} ({zero/n_stable*100:.1f}%)")

    # ── 2. Top 10 stable borrowers with largest positive PD_change ─────────
    print("\n" + "=" * 60)
    print("  2. TOP 10 STABLE BORROWERS WITH LARGEST POSITIVE PD_change")
    print("=" * 60)

    # Build a comparison DataFrame
    comparison = pd.DataFrame(index=range(n_stable))
    for var in COMPARE_VARS:
        comparison[f"{var}_orig"] = stable_orig[var].values
        comparison[f"{var}_curr"] = stable_curr[var].values
        comparison[f"{var}_delta"] = stable_curr[var].values - stable_orig[var].values
    comparison["PD_orig"] = stable_orig["PD"].values
    comparison["PD_curr"] = stable_curr["PD"].values
    comparison["PD_change"] = pd_change

    # Sort by PD_change descending
    top10 = comparison.nlargest(10, "PD_change")

    # Display
    display_cols = ["PD_orig", "PD_curr", "PD_change"]
    for var in COMPARE_VARS:
        display_cols.append(f"{var}_orig")
        display_cols.append(f"{var}_curr")
        display_cols.append(f"{var}_delta")

    # Print a compact table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n  Before/After borrower variables for top 10 largest PD increases:")
    print("-" * 60)

    for idx in top10.index:
        row = top10.loc[idx]
        print(f"\n  Borrower #{idx} (stable trajectory)")
        print(f"    PD_origin : {row['PD_orig']:.4f}")
        print(f"    PD_current: {row['PD_curr']:.4f}")
        print(f"    PD_change : {row['PD_change']:+.4f}")
        print(f"    Variable changes (orig -> curr, delta):")
        for var in COMPARE_VARS:
            o_val = row[f"{var}_orig"]
            c_val = row[f"{var}_curr"]
            d_val = row[f"{var}_delta"]
            marker = " ***" if abs(d_val) > 1e-8 else ""
            print(f"      {var:45s}: {o_val:>12.4f} -> {c_val:>12.4f}  (delta={d_val:+.4f}){marker}")

    # ── 3. Analyze cause of large PD increases ─────────────────────────────
    print("\n" + "=" * 60)
    print("  3. ROOT CAUSE ANALYSIS: Delinquency vs Other Variables")
    print("=" * 60)

    # For the top 10, check which variables changed
    print("\n  Top-10 borrowers: which variables changed?")
    var_change_counts = {}
    for var in COMPARE_VARS:
        delta_col = f"{var}_delta"
        changed = (top10[delta_col].abs() > 1e-8).sum()
        var_change_counts[var] = changed
        print(f"    {var:45s}: changed in {changed}/10 borrowers")

    # Delinquency-specific analysis
    print("\n  Delinquency variable changes in top 10:")
    for var in DELINQUENCY_VARS:
        deltas = top10[f"{var}_delta"].values
        any_change = (np.abs(deltas) > 1e-8).sum()
        increases = (deltas > 1e-8).sum()
        decreases = (deltas < -1e-8).sum()
        print(f"    {var:45s}: {any_change} changed ({increases} up, {decreases} down)")
        print(f"      deltas: {deltas}")

    # Non-delinquency variables
    non_delinq_vars = [v for v in COMPARE_VARS if v not in DELINQUENCY_VARS]
    print(f"\n  Non-delinquency variable changes in top 10:")
    for var in non_delinq_vars:
        deltas = top10[f"{var}_delta"].values
        any_change = (np.abs(deltas) > 1e-8).sum()
        print(f"    {var:45s}: changed in {any_change}/10 borrowers")
        if any_change > 0:
            print(f"      deltas: {deltas}")

    # ── 4. Broader analysis across ALL stable borrowers ────────────────────
    print("\n" + "=" * 60)
    print("  4. BROAD ANALYSIS ACROSS ALL STABLE BORROWERS")
    print("=" * 60)

    # For all stable borrowers, what fraction had delinquency changes?
    print("\n  Variable change frequency across all stable borrowers:")
    for var in COMPARE_VARS:
        delta = stable_curr[var].values - stable_orig[var].values
        changed = (np.abs(delta) > 1e-8).sum()
        pct = changed / n_stable * 100
        print(f"    {var:45s}: {changed:5d} ({pct:5.1f}%)")

    # Delinquency changes specifically
    print("\n  Delinquency changes across all stable borrowers:")
    any_delinq_change = np.zeros(n_stable, dtype=bool)
    for var in DELINQUENCY_VARS:
        delta = stable_curr[var].values - stable_orig[var].values
        any_delinq_change |= (np.abs(delta) > 1e-8)
    print(f"    Any delinquency change: {any_delinq_change.sum()} ({any_delinq_change.mean()*100:.1f}%)")

    # Among stable borrowers with large PD increase (> +0.10), how many had delinq changes?
    large_increase_mask = pd_change > 0.10
    n_large = large_increase_mask.sum()
    print(f"\n  Stable borrowers with PD_change > +0.10: {n_large}")
    if n_large > 0:
        delinq_changed_among_large = any_delinq_change[large_increase_mask].sum()
        print(f"    Of these, delinquency changed: {delinq_changed_among_large} ({delinq_changed_among_large/n_large*100:.1f}%)")

        # What about non-delinquency changes?
        for var in non_delinq_vars:
            delta = stable_curr[var].values - stable_orig[var].values
            changed_large = (np.abs(delta[large_increase_mask]) > 1e-8).sum()
            print(f"    {var:45s}: changed in {changed_large}/{n_large} ({changed_large/n_large*100:.1f}%)")

    # ── 5. Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  5. VERDICT: Is the stable trajectory reasonable?")
    print("=" * 60)

    median_change = np.median(pd_change)
    mean_change = np.mean(pd_change)
    pct_positive = (pd_change > 0).mean() * 100

    # Check if delinquency is the primary driver
    delinq_driven_count = 0
    non_delinq_driven_count = 0
    for i in range(n_stable):
        if abs(pd_change[i]) > 0.02:  # meaningful change
            has_delinq_change = any(
                abs(stable_curr.iloc[i][v] - stable_orig.iloc[i][v]) > 1e-8
                for v in DELINQUENCY_VARS
            )
            has_non_delinq_change = any(
                abs(stable_curr.iloc[i][v] - stable_orig.iloc[i][v]) > 1e-8
                for v in non_delinq_vars
            )
            if has_delinq_change:
                delinq_driven_count += 1
            if has_non_delinq_change:
                non_delinq_driven_count += 1

    meaningful = (np.abs(pd_change) > 0.02).sum()
    print(f"\n  Stable borrowers with meaningful PD_change (|delta| > 0.02): {meaningful}")
    print(f"    Delinquency-driven     : {delinq_driven_count} ({delinq_driven_count/max(meaningful,1)*100:.1f}%)")
    print(f"    Non-delinquency-driven : {non_delinq_driven_count} ({non_delinq_driven_count/max(meaningful,1)*100:.1f}%)")
    print(f"    (A borrower can be in both categories if multiple vars changed)")

    print(f"\n  Summary statistics:")
    print(f"    Median PD_change : {median_change:+.4f}")
    print(f"    Mean PD_change   : {mean_change:+.4f}")
    print(f"    % with positive  : {pct_positive:.1f}%")

    print(f"\n  Assessment:")
    if pct_positive > 80 and median_change > 0.03:
        print(f"    -> The stable trajectory appears SYSTEMATICALLY TOO RISK-INCREASING.")
        print(f"       {pct_positive:.1f}% of stable borrowers see a PD increase, with a")
        print(f"       median shift of {median_change:+.4f}. A 'stable' trajectory should")
        print(f"       show roughly balanced or minimal PD changes.")
    elif pct_positive > 65 and median_change > 0.02:
        print(f"    -> The stable trajectory shows a BIAS toward risk-increase.")
        print(f"       {pct_positive:.1f}% of stable borrowers see a PD increase, with a")
        print(f"       median shift of {median_change:+.4f}. This may be too aggressive")
        print(f"       for a trajectory labelled 'stable'.")
    else:
        print(f"    -> The stable trajectory appears REASONABLE.")
        print(f"       {pct_positive:.1f}% positive with median {median_change:+.4f} is within")
        print(f"       expected bounds for a 'stable' trajectory.")

    print("\n" + "=" * 60)
    print("  Analysis complete.")


if __name__ == "__main__":
    main()