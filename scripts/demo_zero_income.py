"""
Demonstration of the zero-income borrower handling functionality.

This script shows how the updated synthetic borrower snapshot modification module
handles borrowers with zero income at origination.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.modification import construct_current_snapshot, DEFAULT_CONFIG

def demonstrate_zero_income_functionality():
    """Demonstrate the zero-income handling functionality."""
    print("=== Zero-Income Borrower Handling Demonstration ===\n")

    # Create a sample dataset with mixed zero and non-zero income borrowers
    print("1. Creating sample dataset with zero-income borrowers...")
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 0, 0, 0, 0, 0, 0, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.5, 0.6],
        'age': [30, 28, 35, 27, 40, 32, 29, 38],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 0, 2, 1, 0, 3, 1, 0],
        'DebtRatio': [0.3, 0.8, 0.2, 0.9, 0.4, 0.7, 0.3, 0.6],
        'MonthlyIncome': [0, 5000, 0, 6000, 0, 7000, 4000, 0],  # 4 zero-income, 4 non-zero
        'NumberOfOpenCreditLinesAndLoans': [5, 4, 6, 3, 5, 4, 5, 6],
        'NumberOfTimes90DaysLate': [0, 0, 1, 0, 0, 1, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 1, 2, 1, 1, 1, 1, 2],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 0, 1, 0, 0, 1, 0, 0],
        'NumberOfDependents': [2, 1, 3, 0, 2, 1, 1, 2]
    })

    print(f"Original dataset:")
    print(f"- Total borrowers: {len(orig_df)}")
    print(f"- Zero-income borrowers: {(orig_df['MonthlyIncome'] == 0).sum()}")
    print(f"- Non-zero income borrowers: {(orig_df['MonthlyIncome'] > 0).sum()}")
    print(f"- Borrowers with NaN income: {orig_df['MonthlyIncome'].isna().sum()}")
    print()

    # Show original zero-income borrowers
    zero_income_orig = orig_df[orig_df['MonthlyIncome'] == 0]
    print("Original zero-income borrowers:")
    print(zero_income_orig[['MonthlyIncome', 'DebtRatio', 'age']])
    print()

    # Generate current snapshot
    print("2. Generating current snapshot (2 years later)...")
    current_df = construct_current_snapshot(orig_df, seed=42)

    print(f"Current snapshot:")
    print(f"- Total borrowers: {len(current_df)}")
    print(f"- Zero-income borrowers: {(current_df['MonthlyIncome'] == 0).sum()}")
    print(f"- Positive income borrowers: {(current_df['MonthlyIncome'] > 0).sum()}")
    print()

    # Show trajectory distribution
    trajectory_counts = current_df['trajectory'].value_counts()
    print("Trajectory distribution:")
    for trajectory, count in trajectory_counts.items():
        print(f"- {trajectory}: {count} borrowers")
    print()

    # Analyze zero-income transitions by trajectory
    print("3. Zero-income transition analysis by trajectory:")

    for trajectory in ['improving', 'stable', 'deteriorating']:
        trajectory_df = current_df[current_df['trajectory'] == trajectory]
        orig_zero_in_trajectory = orig_df.loc[trajectory_df.index, 'MonthlyIncome'] == 0
        n_orig_zero = orig_zero_in_trajectory.sum()

        if n_orig_zero > 0:
            current_zero = (trajectory_df['MonthlyIncome'] == 0).sum()
            transitioned_to_positive = n_orig_zero - current_zero
            transition_rate = transitioned_to_positive / n_orig_zero

            print(f"  {trajectory.capitalize()} trajectory:")
            print(f"    - Original zero-income borrowers: {n_orig_zero}")
            print(f"    - Remained at zero: {current_zero}")
            print(f"    - Transitioned to positive: {transitioned_to_positive}")
            print(f"    - Transition rate: {transition_rate:.1%}")
        else:
            print(f"  {trajectory.capitalize()} trajectory: No original zero-income borrowers")

    print()

    # Show examples of zero-income borrowers who transitioned to positive income
    orig_zero_mask = orig_df['MonthlyIncome'] == 0
    current_positive_mask = current_df['MonthlyIncome'] > 0
    transitioned_borrowers = current_df[orig_zero_mask & current_positive_mask]

    if len(transitioned_borrowers) > 0:
        print("4. Examples of zero-income borrowers who transitioned to positive income:")
        display_cols = ['MonthlyIncome', 'DebtRatio', 'age', 'trajectory']
        print(transitioned_borrowers[display_cols])
        print()
    else:
        print("4. No zero-income borrowers transitioned to positive income in this run.")
        print()

    # Show examples of zero-income borrowers who remained at zero
    remained_zero_borrowers = current_df[orig_zero_mask & (current_df['MonthlyIncome'] == 0)]

    if len(remained_zero_borrowers) > 0:
        print("5. Examples of zero-income borrowers who remained at zero:")
        display_cols = ['MonthlyIncome', 'DebtRatio', 'age', 'trajectory']
        print(remained_zero_borrowers[display_cols])
        print()
    else:
        print("5. No zero-income borrowers remained at zero in this run.")
        print()

    # Show non-zero income borrowers (should remain non-zero)
    orig_non_zero_mask = orig_df['MonthlyIncome'] > 0
    non_zero_borrowers = current_df[orig_non_zero_mask]

    print("6. Non-zero income borrowers (should all remain non-zero):")
    print(f"   - Count: {len(non_zero_borrowers)}")
    print(f"   - All have positive income: {(non_zero_borrowers['MonthlyIncome'] > 0).all()}")
    print()

    # Show DebtRatio handling for zero-income borrowers
    print("7. DebtRatio handling for zero-income borrowers:")
    orig_zero_indices = orig_df[orig_df['MonthlyIncome'] == 0].index
    zero_income_debt_ratios = current_df.loc[orig_zero_indices, ['DebtRatio', 'trajectory']]
    print("   DebtRatio values for original zero-income borrowers:")
    print(zero_income_debt_ratios)
    print(f"   - All DebtRatios non-negative: {(zero_income_debt_ratios['DebtRatio'] >= 0).all()}")
    print(f"   - Max DebtRatio: {zero_income_debt_ratios['DebtRatio'].max():.2f}")
    print()

    # Demonstrate custom configuration
    print("8. Demonstrating custom zero-income configuration...")

    custom_config = {
        'zero_income_transition': {
            'improving_probability': 0.95,  # Very high chance
            'stable_probability': 0.05,     # Very low chance
            'deteriorating_probability': 0.01, # Almost no chance
            'positive_income_distribution': {
                'mean': 4000,
                'std': 1000,
                'min_value': 2500,
                'max_value': 12000
            }
        }
    }

    current_df_custom = construct_current_snapshot(orig_df, seed=42, config=custom_config)

    # Compare transition rates
    orig_zero_in_custom = orig_df.loc[current_df_custom.index, 'MonthlyIncome'] == 0
    custom_transition_rate = ((current_df_custom['MonthlyIncome'] > 0) & orig_zero_in_custom).sum() / orig_zero_in_custom.sum()

    orig_zero_in_default = orig_df.loc[current_df.index, 'MonthlyIncome'] == 0
    default_transition_rate = ((current_df['MonthlyIncome'] > 0) & orig_zero_in_default).sum() / orig_zero_in_default.sum()

    print(f"   - Default config transition rate: {default_transition_rate:.1%}")
    print(f"   - Custom config transition rate: {custom_transition_rate:.1%}")
    print(f"   - Custom config has higher transition rate as expected: {custom_transition_rate > default_transition_rate}")
    print()

    print("=== Demonstration Complete ===")
    print()
    print("Key features demonstrated:")
    print("[+] Zero-income preservation at origination")
    print("[+] Configurable zero -> positive income transitions")
    print("[+] Trajectory-dependent transition probabilities")
    print("[+] Plausible positive income generation within bounds")
    print("[+] Special DebtRatio handling for zero-income borrowers")
    print("[+] Non-zero income borrowers use existing logic unchanged")
    print("[+] Custom configuration support")

if __name__ == "__main__":
    demonstrate_zero_income_functionality()