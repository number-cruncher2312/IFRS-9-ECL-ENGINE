"""
Focused analysis of DebtRatio transformations for zero-income borrowers.

This script inspects the before/after distribution of DebtRatio specifically
for zero-income borrowers to ensure we're not creating bizarre current states.
"""

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add the project root to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.modification import construct_current_snapshot, DEFAULT_CONFIG

def analyze_debt_ratio_distributions():
    """Analyze DebtRatio distributions for zero-income borrowers."""
    print("=== DebtRatio Analysis for Zero-Income Borrowers ===\n")

    # Create test dataset with zero-income borrowers and various DebtRatio values
    # Include some extreme cases to test the transformation
    np.random.seed(42)

    # Generate realistic DebtRatio values including some high ones
    debt_ratios = [
        0.1, 0.2, 0.3, 0.4, 0.5,  # Normal range
        0.6, 0.7, 0.8, 0.9,        # High but reasonable
        1.0, 1.2, 1.5, 1.8,        # Very high
        2.0, 2.5, 3.0,             # Extreme cases
        0.05, 0.08                 # Very low
    ]

    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0] * len(debt_ratios),
        'RevolvingUtilizationOfUnsecuredLines': [0.5] * len(debt_ratios),
        'age': [30] * len(debt_ratios),
        'NumberOfTime30-59DaysPastDueNotWorse': [1] * len(debt_ratios),
        'DebtRatio': debt_ratios,
        'MonthlyIncome': [0] * len(debt_ratios),  # All zero-income
        'NumberOfOpenCreditLinesAndLoans': [5] * len(debt_ratios),
        'NumberOfTimes90DaysLate': [0] * len(debt_ratios),
        'NumberRealEstateLoansOrLines': [1] * len(debt_ratios),
        'NumberOfTime60-89DaysPastDueNotWorse': [0] * len(debt_ratios),
        'NumberOfDependents': [2] * len(debt_ratios)
    })

    print(f"Original zero-income borrowers: {len(orig_df)}")
    print(f"DebtRatio range: {orig_df['DebtRatio'].min():.2f} to {orig_df['DebtRatio'].max():.2f}")
    print()

    # Show original DebtRatio distribution
    print("Original DebtRatio values for zero-income borrowers:")
    print(orig_df[['DebtRatio']].describe())
    print()

    # Generate current snapshot
    current_df = construct_current_snapshot(orig_df, seed=42)

    # Extract zero-income borrowers from current snapshot
    # (some may have transitioned to positive income, so we need to identify original zero-income)
    orig_zero_indices = orig_df[orig_df['MonthlyIncome'] == 0].index
    zero_income_current = current_df.loc[orig_zero_indices]

    print("Current DebtRatio values for original zero-income borrowers:")
    print(zero_income_current[['DebtRatio', 'trajectory']].describe())
    print()

    # Compare before/after
    comparison_df = pd.DataFrame({
        'Original_DebtRatio': orig_df.loc[orig_zero_indices, 'DebtRatio'],
        'Current_DebtRatio': zero_income_current['DebtRatio'],
        'Trajectory': zero_income_current['trajectory'],
        'Change': zero_income_current['DebtRatio'] - orig_df.loc[orig_zero_indices, 'DebtRatio']
    })

    print("Before/After Comparison:")
    print(comparison_df[['Original_DebtRatio', 'Current_DebtRatio', 'Trajectory', 'Change']])
    print()

    # Analyze by trajectory
    print("DebtRatio Changes by Trajectory:")
    for trajectory in ['improving', 'stable', 'deteriorating']:
        trajectory_df = comparison_df[comparison_df['Trajectory'] == trajectory]
        if len(trajectory_df) > 0:
            orig_mean = trajectory_df['Original_DebtRatio'].mean()
            current_mean = trajectory_df['Current_DebtRatio'].mean()
            change_mean = trajectory_df['Change'].mean()

            print(f"\n{trajectory.capitalize()} Trajectory ({len(trajectory_df)} borrowers):")
            print(f"  Original mean: {orig_mean:.3f}")
            print(f"  Current mean:  {current_mean:.3f}")
            print(f"  Mean change:   {change_mean:.3f}")
            print(f"  Max change:    {trajectory_df['Change'].max():.3f}")
            print(f"  Min change:    {trajectory_df['Change'].min():.3f}")

            # Check for problematic transformations
            extreme_increase = (trajectory_df['Change'] > 0.5).sum()
            extreme_decrease = (trajectory_df['Change'] < -0.5).sum()
            print(f"  Extreme increases (>0.5): {extreme_increase}")
            print(f"  Extreme decreases (<-0.5): {extreme_decrease}")

    print()

    # Check for potential issues
    print("=== Potential Issue Analysis ===")

    # 1. Check if we're creating negative DebtRatios
    negative_debt = (zero_income_current['DebtRatio'] < 0).sum()
    print(f"Negative DebtRatios created: {negative_debt}")

    # 2. Check for unreasonable increases
    large_increases = ((zero_income_current['DebtRatio'] - orig_df.loc[orig_zero_indices, 'DebtRatio']) > 1.0).sum()
    print(f"Large increases (>1.0): {large_increases}")

    # 3. Check if high DebtRatios are being made worse
    high_debt_orig = orig_df.loc[orig_zero_indices, 'DebtRatio'] > 1.0
    high_debt_current = zero_income_current['DebtRatio'] > 1.0
    high_debt_worsened = ((high_debt_orig) & (high_debt_current) &
                         (zero_income_current['DebtRatio'] > orig_df.loc[orig_zero_indices, 'DebtRatio'])).sum()
    print(f"High DebtRatios (>1.0) that worsened: {high_debt_worsened}")

    # 4. Check distribution characteristics
    print(f"\nDistribution Characteristics:")
    print(f"Original skewness: {orig_df.loc[orig_zero_indices, 'DebtRatio'].skew():.3f}")
    print(f"Current skewness:  {zero_income_current['DebtRatio'].skew():.3f}")

    # 5. Check if we're creating bizarre states
    bizarre_states = []
    for i, row in comparison_df.iterrows():
        if (row['Original_DebtRatio'] > 2.0 and row['Current_DebtRatio'] > 3.0):
            bizarre_states.append(f"ID {i}: {row['Original_DebtRatio']:.2f} -> {row['Current_DebtRatio']:.2f}")
        elif (row['Original_DebtRatio'] < 0.5 and row['Current_DebtRatio'] > 2.0):
            bizarre_states.append(f"ID {i}: {row['Original_DebtRatio']:.2f} -> {row['Current_DebtRatio']:.2f}")

    if bizarre_states:
        print(f"\nPotential bizarre transformations:")
        for state in bizarre_states:
            print(f"  {state}")
    else:
        print(f"\nNo obvious bizarre transformations detected.")

    # Visualization
    try:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(orig_df.loc[orig_zero_indices, 'DebtRatio'], bins=10, color='blue', alpha=0.7)
        plt.title('Original DebtRatio Distribution')
        plt.xlabel('DebtRatio')
        plt.ylabel('Count')

        plt.subplot(1, 2, 2)
        plt.hist(zero_income_current['DebtRatio'], bins=10, color='green', alpha=0.7)
        plt.title('Current DebtRatio Distribution')
        plt.xlabel('DebtRatio')
        plt.ylabel('Count')

        plt.tight_layout()
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs'), exist_ok=True)
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'debt_ratio_distribution_comparison.png')
        plt.savefig(output_path)
        print(f"\nSaved distribution comparison plot to '{output_path}'")

    except Exception as e:
        print(f"\nCould not generate plot: {e}")

    return comparison_df

def test_alternative_approaches():
    """Test alternative approaches for DebtRatio handling."""
    print("\n=== Testing Alternative DebtRatio Approaches ===\n")

    # Create test dataset with extreme DebtRatios
    extreme_debt_ratios = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0] * len(extreme_debt_ratios),
        'RevolvingUtilizationOfUnsecuredLines': [0.5] * len(extreme_debt_ratios),
        'age': [30] * len(extreme_debt_ratios),
        'NumberOfTime30-59DaysPastDueNotWorse': [1] * len(extreme_debt_ratios),
        'DebtRatio': extreme_debt_ratios,
        'MonthlyIncome': [0] * len(extreme_debt_ratios),
        'NumberOfOpenCreditLinesAndLoans': [5] * len(extreme_debt_ratios),
        'NumberOfTimes90DaysLate': [0] * len(extreme_debt_ratios),
        'NumberRealEstateLoansOrLines': [1] * len(extreme_debt_ratios),
        'NumberOfTime60-89DaysPastDueNotWorse': [0] * len(extreme_debt_ratios),
        'NumberOfDependents': [2] * len(extreme_debt_ratios)
    })

    print("Testing extreme DebtRatio cases:")
    print("Original values:", extreme_debt_ratios)

    # Test current approach
    current_df = construct_current_snapshot(orig_df, seed=42)
    orig_zero_indices = orig_df[orig_df['MonthlyIncome'] == 0].index
    current_values = current_df.loc[orig_zero_indices, 'DebtRatio'].values

    print("Current approach results:", [f"{val:.3f}" for val in current_values])

    # Analyze the transformations
    changes = current_values - extreme_debt_ratios
    print("Changes:", [f"{change:.3f}" for change in changes])

    # Check for problematic cases
    problematic = []
    for i, (orig, curr, change) in enumerate(zip(extreme_debt_ratios, current_values, changes)):
        if orig > 2.0 and curr > orig + 0.5:  # High debt getting significantly worse
            problematic.append(f"ID {i}: {orig} -> {curr} (+{change:.2f})")
        elif orig > 1.5 and curr > 3.0:  # Crossing into very high territory
            problematic.append(f"ID {i}: {orig} -> {curr} (+{change:.2f})")

    if problematic:
        print("\nPotentially problematic transformations:")
        for case in problematic:
            print(f"  {case}")
    else:
        print("\nNo obviously problematic transformations with current approach.")

def suggest_improvements():
    """Suggest potential improvements based on analysis."""
    print("\n=== Suggested Improvements ===\n")

    print("Current approach analysis:")
    print("- Uses additive shocks instead of multiplicative")
    print("- Trajectory-dependent parameters:")
    print("  * Improving: mean=-0.2, std=0.1 (tends to reduce debt)")
    print("  * Stable: mean=0.0, std=0.05 (small changes)")
    print("  * Deteriorating: mean=0.1, std=0.1 (tends to increase debt)")
    print("- Bounds: min_value=0.0 (no negative debt ratios)")

    print("\nPotential improvement options:")

    print("\n1. Cap extreme transformations:")
    print("   - Limit maximum absolute change for high DebtRatios")
    print("   - Example: if DebtRatio > 1.5, cap change to ±0.3")

    print("\n2. Make improvements more conservative for high DebtRatios:")
    print("   - Reduce shock magnitude for DebtRatio > 1.0")
    print("   - Could use a scaling factor based on original DebtRatio")

    print("\n3. Add maximum DebtRatio bound:")
    print("   - Set a reasonable upper limit (e.g., max_value=3.0 or 4.0)")
    print("   - Prevents runaway debt ratios")

    print("\n4. Different approach for very high DebtRatios:")
    print("   - For DebtRatio > 2.0, use different parameters")
    print("   - Could make deteriorating trajectory less aggressive")

    print("\n5. Consider no change for extreme cases:")
    print("   - For DebtRatio > 2.5, apply minimal or no transformation")
    print("   - Preserves the extreme state without making it worse")

    print("\nRecommendation:")
    print("The current approach is directionally reasonable but could benefit from:")
    print("1. Adding an upper bound to prevent unreasonable DebtRatios")
    print("2. Making transformations more conservative for high DebtRatios")
    print("3. Ensuring we don't significantly worsen already high DebtRatios")

if __name__ == "__main__":
    # Run the analysis
    comparison_df = analyze_debt_ratio_distributions()
    test_alternative_approaches()
    suggest_improvements()

    print("\n=== Analysis Complete ===")
    print("Review the results to determine if the current DebtRatio handling")
    print("for zero-income borrowers is appropriate or needs adjustment.")