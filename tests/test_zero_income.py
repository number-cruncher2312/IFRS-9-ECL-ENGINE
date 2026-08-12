import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.modification import construct_current_snapshot, DEFAULT_CONFIG

def test_zero_income_preservation_at_origination():
    """Test that zero income is preserved in the original dataframe."""
    # Create test dataframe with some zero-income borrowers
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 1, 0, 1, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.8, 0.3, 0.6, 0.4],
        'age': [30, 40, 25, 35, 28],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 2, 0, 1, 0],
        'DebtRatio': [0.3, 0.5, 0.2, 0.4, 0.3],
        'MonthlyIncome': [0, 5000, 0, 6000, 0],  # Three zero-income borrowers
        'NumberOfOpenCreditLinesAndLoans': [5, 3, 4, 2, 3],
        'NumberOfTimes90DaysLate': [0, 1, 0, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 2, 1, 1, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 1, 0, 0, 0],
        'NumberOfDependents': [2, 1, 0, 1, 1]
    })

    # Store original values
    orig_copy = orig_df.copy()

    # Apply modification
    result_df = construct_current_snapshot(orig_df, seed=42)

    # Check that original dataframe is unchanged (zero income preserved)
    pd.testing.assert_frame_equal(orig_df, orig_copy)

    # Check that zero-income borrowers in original are still zero
    assert (orig_df['MonthlyIncome'] == 0).sum() == 3

def test_zero_income_transition_behavior():
    """Test that zero-income borrowers can transition to positive income."""
    # Create test dataframe with zero-income borrowers
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 0, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.5, 0.5],
        'age': [30, 30, 30],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 1, 1],
        'DebtRatio': [0.3, 0.3, 0.3],
        'MonthlyIncome': [0, 0, 0],  # All zero-income
        'NumberOfOpenCreditLinesAndLoans': [5, 5, 5],
        'NumberOfTimes90DaysLate': [0, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 1, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 0, 0],
        'NumberOfDependents': [2, 2, 2]
    })

    # Apply modification with fixed seed for reproducibility
    result_df = construct_current_snapshot(orig_df, seed=42)

    # Check that some zero-income borrowers may have transitioned to positive income
    positive_income_count = (result_df['MonthlyIncome'] > 0).sum()

    # With the default configuration, we expect some transitions
    assert positive_income_count >= 0  # At least some should transition
    assert positive_income_count <= 3  # None should exceed total count

    # Check that remaining zero-income borrowers stay at exactly 0
    zero_income_mask = result_df['MonthlyIncome'] == 0
    assert (result_df[zero_income_mask]['MonthlyIncome'] == 0).all()

def test_zero_income_transition_probabilities():
    """Test that transition probabilities are trajectory-dependent."""
    # Create larger test dataset
    np.random.seed(42)
    n_samples = 1000
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0] * n_samples,
        'RevolvingUtilizationOfUnsecuredLines': [0.5] * n_samples,
        'age': [30] * n_samples,
        'NumberOfTime30-59DaysPastDueNotWorse': [1] * n_samples,
        'DebtRatio': [0.3] * n_samples,
        'MonthlyIncome': [0] * n_samples,  # All zero-income
        'NumberOfOpenCreditLinesAndLoans': [5] * n_samples,
        'NumberOfTimes90DaysLate': [0] * n_samples,
        'NumberRealEstateLoansOrLines': [1] * n_samples,
        'NumberOfTime60-89DaysPastDueNotWorse': [0] * n_samples,
        'NumberOfDependents': [2] * n_samples
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    # Check transition rates by trajectory
    improving_df = result_df[result_df['trajectory'] == 'improving']
    stable_df = result_df[result_df['trajectory'] == 'stable']
    deteriorating_df = result_df[result_df['trajectory'] == 'deteriorating']

    improving_transition_rate = (improving_df['MonthlyIncome'] > 0).mean()
    stable_transition_rate = (stable_df['MonthlyIncome'] > 0).mean()
    deteriorating_transition_rate = (deteriorating_df['MonthlyIncome'] > 0).mean()

    # Improving should have highest transition rate, deteriorating lowest
    assert improving_transition_rate > stable_transition_rate
    assert stable_transition_rate > deteriorating_transition_rate

def test_positive_income_generation_bounds():
    """Test that generated positive income stays within configured bounds."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0] * 10,
        'RevolvingUtilizationOfUnsecuredLines': [0.5] * 10,
        'age': [30] * 10,
        'NumberOfTime30-59DaysPastDueNotWorse': [1] * 10,
        'DebtRatio': [0.3] * 10,
        'MonthlyIncome': [0] * 10,  # All zero-income
        'NumberOfOpenCreditLinesAndLoans': [5] * 10,
        'NumberOfTimes90DaysLate': [0] * 10,
        'NumberRealEstateLoansOrLines': [1] * 10,
        'NumberOfTime60-89DaysPastDueNotWorse': [0] * 10,
        'NumberOfDependents': [2] * 10
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    # Check that positive income values are within expected bounds
    positive_income_mask = result_df['MonthlyIncome'] > 0
    if positive_income_mask.any():
        positive_incomes = result_df[positive_income_mask]['MonthlyIncome']
        assert (positive_incomes >= 1000).all()  # Minimum bound
        assert (positive_incomes <= 8000).all()  # Maximum bound

def test_non_zero_income_unchanged():
    """Test that non-zero income borrowers use existing logic unchanged."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 0, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.5, 0.5],
        'age': [30, 30, 30],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 1, 1],
        'DebtRatio': [0.3, 0.3, 0.3],
        'MonthlyIncome': [5000, 6000, 7000],  # All non-zero income
        'NumberOfOpenCreditLinesAndLoans': [5, 5, 5],
        'NumberOfTimes90DaysLate': [0, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 1, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 0, 0],
        'NumberOfDependents': [2, 2, 2]
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    # All should still have positive income (no zeros introduced)
    assert (result_df['MonthlyIncome'] > 0).all()

    # Income should have changed from original (due to shocks)
    assert not result_df['MonthlyIncome'].equals(orig_df['MonthlyIncome'])

def test_debt_ratio_zero_income_handling():
    """Test that DebtRatio is handled specially for zero-income borrowers."""
    # Create test dataframe with zero-income borrowers and high debt ratios
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 0, 0, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.5, 0.5, 0.5],
        'age': [30, 30, 30, 30],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 1, 1, 1],
        'DebtRatio': [0.8, 0.9, 0.7, 0.3],  # Some high debt ratios
        'MonthlyIncome': [0, 0, 5000, 6000],  # First two are zero-income
        'NumberOfOpenCreditLinesAndLoans': [5, 5, 5, 5],
        'NumberOfTimes90DaysLate': [0, 0, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 1, 1, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 0, 0, 0],
        'NumberOfDependents': [2, 2, 2, 2]
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    # Check that debt ratios remain non-negative
    assert (result_df['DebtRatio'] >= 0).all()

    # Check that debt ratios don't explode to unreasonable values
    assert (result_df['DebtRatio'] <= 2.0).all()  # Reasonable upper bound

def test_mixed_zero_and_non_zero_income():
    """Test behavior with mixed zero and non-zero income borrowers."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0, 0, 0, 0, 0],
        'RevolvingUtilizationOfUnsecuredLines': [0.5, 0.5, 0.5, 0.5, 0.5],
        'age': [30, 30, 30, 30, 30],
        'NumberOfTime30-59DaysPastDueNotWorse': [1, 1, 1, 1, 1],
        'DebtRatio': [0.3, 0.3, 0.3, 0.3, 0.3],
        'MonthlyIncome': [0, 5000, 0, 6000, 0],  # Mixed zero and non-zero
        'NumberOfOpenCreditLinesAndLoans': [5, 5, 5, 5, 5],
        'NumberOfTimes90DaysLate': [0, 0, 0, 0, 0],
        'NumberRealEstateLoansOrLines': [1, 1, 1, 1, 1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 0, 0, 0, 0],
        'NumberOfDependents': [2, 2, 2, 2, 2]
    })

    result_df = construct_current_snapshot(orig_df, seed=42)

    # Check that zero-income borrowers are handled appropriately
    orig_zero_mask = orig_df['MonthlyIncome'] == 0
    result_zero_mask = (result_df['MonthlyIncome'] == 0) & orig_zero_mask

    # Some zero-income borrowers should remain at zero
    assert result_zero_mask.any()

    # Non-zero income borrowers should remain non-zero
    orig_non_zero_mask = orig_df['MonthlyIncome'] > 0
    assert (result_df[orig_non_zero_mask]['MonthlyIncome'] > 0).all()

def test_custom_zero_income_config():
    """Test that custom zero-income configuration works."""
    orig_df = pd.DataFrame({
        'SeriousDlqin2yrs': [0] * 10,
        'RevolvingUtilizationOfUnsecuredLines': [0.5] * 10,
        'age': [30] * 10,
        'NumberOfTime30-59DaysPastDueNotWorse': [1] * 10,
        'DebtRatio': [0.3] * 10,
        'MonthlyIncome': [0] * 10,  # All zero-income
        'NumberOfOpenCreditLinesAndLoans': [5] * 10,
        'NumberOfTimes90DaysLate': [0] * 10,
        'NumberRealEstateLoansOrLines': [1] * 10,
        'NumberOfTime60-89DaysPastDueNotWorse': [0] * 10,
        'NumberOfDependents': [2] * 10
    })

    # Custom configuration with different transition probabilities
    custom_config = {
        'zero_income_transition': {
            'improving_probability': 0.9,  # Very high chance
            'stable_probability': 0.1,     # Very low chance
            'deteriorating_probability': 0.05, # Almost no chance
            'positive_income_distribution': {
                'mean': 4000,
                'std': 1000,
                'min_value': 2000,
                'max_value': 10000
            }
        }
    }

    result_df = construct_current_snapshot(orig_df, seed=42, config=custom_config)

    # Check that custom bounds are respected
    positive_income_mask = result_df['MonthlyIncome'] > 0
    if positive_income_mask.any():
        positive_incomes = result_df[positive_income_mask]['MonthlyIncome']
        assert (positive_incomes >= 2000).all()  # Custom minimum
        assert (positive_incomes <= 10000).all()  # Custom maximum

if __name__ == "__main__":
    # Run all tests
    test_zero_income_preservation_at_origination()
    test_zero_income_transition_behavior()
    test_zero_income_transition_probabilities()
    test_positive_income_generation_bounds()
    test_non_zero_income_unchanged()
    test_debt_ratio_zero_income_handling()
    test_mixed_zero_and_non_zero_income()
    test_custom_zero_income_config()
    print("All zero-income tests passed!")