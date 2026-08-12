import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.modification import construct_current_snapshot, DEFAULT_CONFIG

def test_input_validation():
    """Test that input validation works correctly."""
    # Test with missing required column
    incomplete_df = pd.DataFrame({
        'age': [30, 40],
        'MonthlyIncome': [5000, 6000]
        # Missing other required columns
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        construct_current_snapshot(incomplete_df)

def test_dataframe_not_mutated():
    """Test that the original dataframe is not mutated."""
    # Create test dataframe
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.8],
        'age': [30, 40],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 2],
        'DebtRatio': [0.3, 0.5],
        'MonthlyIncome': [5000, 6000],
        'NumberOfOpenCreditLinesAndLoans': [5, 3],
        'NumberOfTimes90DaysLate': [0, 1],
        'NumberRealEstateLoansOrLines': [1, 2],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1],
        'NumberOfDependents': [2, 1]
    })

    # Store original values
    orig_copy = orig_df.copy()

    # Apply modification
    result_df = construct_current_snapshot(orig_df, seed=42)

    # Check that original dataframe is unchanged
    pd.testing.assert_frame_equal(orig_df, orig_copy)

def test_row_count_preserved():
    """Test that row count is preserved."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.8, 0.3],
        'age': [30, 40, 25],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 2, 0],
        'DebtRatio': [0.3, 0.5, 0.2],
        'MonthlyIncome': [5000, 6000, 4000],
        'NumberOfOpenCreditLinesAndLoans': [5, 3, 4],
        'NumberOfTimes90DaysLate': [0, 1, 0],
        'NumberRealEstateLoansOrLines': [1, 2, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1, 0],
        'NumberOfDependents': [2, 1, 0]
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    assert len(result_df) == len(orig_df)

def test_row_identity_preserved():
    """Test that row identity/order is preserved."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.8, 0.3],
        'age': [30, 40, 25],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 2, 0],
        'DebtRatio': [0.3, 0.5, 0.2],
        'MonthlyIncome': [5000, 6000, 4000],
        'NumberOfOpenCreditLinesAndLoans': [5, 3, 4],
        'NumberOfTimes90DaysLate': [0, 1, 0],
        'NumberRealEstateLoansOrLines': [1, 2, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1, 0],
        'NumberOfDependents': [2, 1, 0]
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    # Check that index is preserved
    pd.testing.assert_index_equal(result_df.index, orig_df.index)

def test_age_increases_by_two():
    """Test that age increases exactly by 2 years."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.8],
        'age': [30, 40],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 2],
        'DebtRatio': [0.3, 0.5],
        'MonthlyIncome': [5000, 6000],
        'NumberOfOpenCreditLinesAndLoans': [5, 3],
        'NumberOfTimes90DaysLate': [0, 1],
        'NumberRealEstateLoansOrLines': [1, 2],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1],
        'NumberOfDependents': [2, 1]
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    expected_ages = orig_df['age'] + 2
    pd.testing.assert_series_equal(result_df['age'], expected_ages)

def test_trajectory_labels_valid():
    """Test that trajectory labels are valid."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1, 0, 1],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.8, 0.3, 0.6],
        'age': [30, 40, 25, 35],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 2, 0, 1],
        'DebtRatio': [0.3, 0.5, 0.2, 0.4],
        'MonthlyIncome': [5000, 6000, 4000, 5500],
        'NumberOfOpenCreditLinesAndLoans': [5, 3, 4, 2],
        'NumberOfTimes90DaysLate': [0, 1, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 2, 1, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1, 0, 0],
        'NumberOfDependents': [2, 1, 0, 1]
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    valid_trajectories = {'stable', 'improving', 'deteriorating'}
    assert set(result_df['trajectory'].unique()).issubset(valid_trajectories)

def test_continuous_variables_in_valid_domains():
    """Test that continuous variables stay within valid domains."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.8],
        'age': [30, 40],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 2],
        'DebtRatio': [0.3, 0.5],
        'MonthlyIncome': [5000, 6000],
        'NumberOfOpenCreditLinesAndLoans': [5, 3],
        'NumberOfTimes90DaysLate': [0, 1],
        'NumberRealEstateLoansOrLines': [1, 2],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1],
        'NumberOfDependents': [2, 1]
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    # Test specific continuous variables
    assert (result_df['MonthlyIncome'] >= 0).all()
    assert (result_df['DebtRatio'] >= 0).all()
    assert (result_df['RevolvingUtilizationOfUnsecuredLines'] >= 0).all()
    assert (result_df['RevolvingUtilizationOfUnsecuredLines'] <= 1.0).all()

def test_count_variables_non_negative_integers():
    """Test that count variables remain non-negative integers."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.8],
        'age': [30, 40],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 2],
        'DebtRatio': [0.3, 0.5],
        'MonthlyIncome': [5000, 6000],
        'NumberOfOpenCreditLinesAndLoans': [5, 3],
        'NumberOfTimes90DaysLate': [0, 1],
        'NumberRealEstateLoansOrLines': [1, 2],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1],
        'NumberOfDependents': [2, 1]
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    count_vars = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberRealEstateLoansOrLines'
    ]

    for var in count_vars:
        assert (result_df[var] >= 0).all()
        assert (result_df[var] % 1 == 0).all()  # Check integer values

def test_reproducibility_same_seed():
    """Test that same seed produces identical output."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1, 0, 1, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.8, 0.3, 0.6, 0.4],
        'age': [30, 40, 25, 35, 28],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 2, 0, 1, 0],
        'DebtRatio': [0.3, 0.5, 0.2, 0.4, 0.3],
        'MonthlyIncome': [5000, 6000, 4000, 5500, 4800],
        'NumberOfOpenCreditLinesAndLoans': [5, 3, 4, 2, 3],
        'NumberOfTimes90DaysLate': [0, 1, 0, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 2, 1, 1, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1, 0, 0, 0],
        'NumberOfDependents': [2, 1, 0, 1, 1]
    })

    result1 = construct_current_snapshot(orig_df.copy(), seed=42)
    result2 = construct_current_snapshot(orig_df.copy(), seed=42)

    pd.testing.assert_frame_equal(result1, result2)

def test_different_seeds_produce_different_results():
    """Test that different seeds can produce different output."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1, 0, 1, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.8, 0.3, 0.6, 0.4],
        'age': [30, 40, 25, 35, 28],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 2, 0, 1, 0],
        'DebtRatio': [0.3, 0.5, 0.2, 0.4, 0.3],
        'MonthlyIncome': [5000, 6000, 4000, 5500, 4800],
        'NumberOfOpenCreditLinesAndLoans': [5, 3, 4, 2, 3],
        'NumberOfTimes90DaysLate': [0, 1, 0, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 2, 1, 1, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1, 0, 0, 0],
        'NumberOfDependents': [2, 1, 0, 1, 1]
    })

    result1 = construct_current_snapshot(orig_df.copy(), seed=42)
    result2 = construct_current_snapshot(orig_df.copy(), seed=123)

    # Results should be different (with high probability)
    assert not result1.equals(result2)

def test_all_trajectories_possible():
    """Test that all three trajectories are possible."""
    # Use a larger dataset to ensure all trajectories appear
    np.random.seed(42)
    n_samples = 1000
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': np.random.randint(0, 2, n_samples),
        'RevolvingUtilizationOfUnsecuredLines': np.random.uniform(0, 1, n_samples),
        'age': np.random.randint(25, 65, n_samples),
        'NumberOfTime30-59DaysPastDueNotWorse': np.random.randint(0, 5, n_samples),
        'DebtRatio': np.random.uniform(0, 1, n_samples),
        'MonthlyIncome': np.random.randint(2000, 10000, n_samples),
        'NumberOfOpenCreditLinesAndLoans': np.random.randint(1, 10, n_samples),
        'NumberOfTimes90DaysLate': np.random.randint(0, 3, n_samples),
        'NumberRealEstateLoansOrLines': np.random.randint(0, 3, n_samples),
        'NumberOfTime60-89DaysPastDueNotWorse': np.random.randint(0, 2, n_samples),
        'NumberOfDependents': np.random.randint(0, 4, n_samples)
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    unique_trajectories = result_df['trajectory'].unique()
    expected_trajectories = {'stable', 'improving', 'deteriorating'}

    assert set(unique_trajectories) == expected_trajectories

def test_stable_trajectory_centered_around_zero():
    """Test that stable trajectory has changes centered approximately around zero."""
    # Create a test dataset
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0] * 100,
        'RevolvingUtilizationOfUnsecuredLines': [0.5] * 100,
        'age': [30] * 100,
        'NumberOfTime30-59DaysPastDueNotWorse': [1] * 100,
        'DebtRatio': [0.3] * 100,
        'MonthlyIncome': [5000] * 100,
        'NumberOfOpenCreditLinesAndLoans': [5] * 100,
        'NumberOfTimes90DaysLate': [0] * 100,
        'NumberRealEstateLoansOrLines': [1] * 100,
        'NumberOfTime60-89DaysPastDueNotWorse': [0] * 100,
        'NumberOfDependents': [2] * 100
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    # Filter for stable trajectory
    stable_df = result_df[result_df['trajectory'] == 'stable']

    # Check that changes are centered around original values
    # For continuous variables
    assert abs(stable_df['MonthlyIncome'].mean() - 5000) < 500  # Within reasonable range
    assert abs(stable_df['DebtRatio'].mean() - 0.3) < 0.1
    assert abs(stable_df['RevolvingUtilizationOfUnsecuredLines'].mean() - 0.5) < 0.1

def test_improving_deteriorating_directional_tendencies():
    """Test that improving/deteriorating trajectories show intended directional tendencies."""
    # Create a test dataset
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0] * 300,
        'RevolvingUtilizationOfUnsecuredLines': [0.5] * 300,
        'age': [30] * 300,
        'NumberOfTime30-59DaysPastDueNotWorse': [2] * 300,
        'DebtRatio': [0.4] * 300,
        'MonthlyIncome': [5000] * 300,
        'NumberOfOpenCreditLinesAndLoans': [5] * 300,
        'NumberOfTimes90DaysLate': [1] * 300,
        'NumberRealEstateLoansOrLines': [1] * 300,
        'NumberOfTime60-89DaysPastDueNotWorse': [1] * 300,
        'NumberOfDependents': [2] * 300
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    # Separate by trajectory
    improving_df = result_df[result_df['trajectory'] == 'improving']
    deteriorating_df = result_df[result_df['trajectory'] == 'deteriorating']

    # Test directional tendencies for income (higher = improvement)
    assert improving_df['MonthlyIncome'].mean() > deteriorating_df['MonthlyIncome'].mean()

    # Test directional tendencies for utilization (lower = improvement)
    assert improving_df['RevolvingUtilizationOfUnsecuredLines'].mean() < deteriorating_df['RevolvingUtilizationOfUnsecuredLines'].mean()

    # Test directional tendencies for debt ratio (lower = improvement)
    assert improving_df['DebtRatio'].mean() < deteriorating_df['DebtRatio'].mean()

    # Test directional tendencies for delinquency counts (lower = improvement)
    assert improving_df['NumberOfTime30-59DaysPastDueNotWorse'].mean() < deteriorating_df['NumberOfTime30-59DaysPastDueNotWorse'].mean()
    assert improving_df['NumberOfTimes90DaysLate'].mean() < deteriorating_df['NumberOfTimes90DaysLate'].mean()

def test_custom_trajectory_probabilities():
    """Test that custom trajectory probabilities work."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0] * 100,
        'RevolvingUtilizationOfUnsecuredLines': [0.5] * 100,
        'age': [30] * 100,
        'NumberOfTime30-59DaysPastDueNotWorse': [1] * 100,
        'DebtRatio': [0.3] * 100,
        'MonthlyIncome': [5000] * 100,
        'NumberOfOpenCreditLinesAndLoans': [5] * 100,
        'NumberOfTimes90DaysLate': [0] * 100,
        'NumberRealEstateLoansOrLines': [1] * 100,
        'NumberOfTime60-89DaysPastDueNotWorse': [0] * 100,
        'NumberOfDependents': [2] * 100
    })

    custom_probs = {'stable': 0.3, 'improving': 0.5, 'deteriorating': 0.2}
    result_df = construct_current_snapshot(orig_df, seed=42, trajectory_probs=custom_probs)

    # Check that trajectory distribution approximately matches custom probabilities
    trajectory_counts = result_df['trajectory'].value_counts(normalize=True)

    assert abs(trajectory_counts.get('stable', 0) - 0.3) < 0.1
    assert abs(trajectory_counts.get('improving', 0) - 0.5) < 0.1
    assert abs(trajectory_counts.get('deteriorating', 0) - 0.2) < 0.1

def test_serious_dlqin_2yrs_preserved():
    """Test that SeriousDlqin2yrs is preserved but not modified."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1, 0, 1, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.8, 0.3, 0.6, 0.4],
        'age': [30, 40, 25, 35, 28],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 2, 0, 1, 0],
        'DebtRatio': [0.3, 0.5, 0.2, 0.4, 0.3],
        'MonthlyIncome': [5000, 6000, 4000, 5500, 4800],
        'NumberOfOpenCreditLinesAndLoans': [5, 3, 4, 2, 3],
        'NumberOfTimes90DaysLate': [0, 1, 0, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 2, 1, 1, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1, 0, 0, 0],
        'NumberOfDependents': [2, 1, 0, 1, 1]
    })

    result_df = construct_current_snapshot(orig_df.copy(), seed=42)

    # SeriousDlqin2yrs should be identical in both dataframes
    pd.testing.assert_series_equal(result_df['SeriousDlqin2yrs'], orig_df['SeriousDlqin2yrs'])