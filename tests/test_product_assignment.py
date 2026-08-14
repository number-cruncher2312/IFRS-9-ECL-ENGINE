import pandas as pd
import numpy as np
import pytest
import sys
import os

from synthetic_data.product_assignment import (
    assign_products,
    assign_products_to_dataframe,
    validate_product_probabilities,
    validate_product_distribution,
    get_default_product_probabilities,
    get_v1_active_products,
    get_v1_inactive_products,
    set_default_product_probabilities,
    DEFAULT_PRODUCT_PROBABILITIES,
    V1_ACTIVE_PRODUCT_PROBABILITIES,
    V1_INACTIVE_PRODUCTS
)
from synthetic_data.product_taxonomy import PRODUCT_TYPES

def test_default_probabilities_correct():
    """Test that default probabilities match the specified requirements."""
    expected_probs = {
        "credit_card": 0.7482,
        "auto_loan": 0.1280,
        "mortgage": 0.1238
    }

    # Check that default probabilities match exactly
    assert DEFAULT_PRODUCT_PROBABILITIES == expected_probs

def test_default_probabilities_sum_to_one():
    """Test that default probabilities sum to 1 within floating-point tolerance."""
    total = sum(DEFAULT_PRODUCT_PROBABILITIES.values())
    assert np.isclose(total, 1.0, atol=1e-6)

def test_get_default_product_probabilities():
    """Test that get_default_product_probabilities returns a copy of defaults."""
    default_probs = get_default_product_probabilities()

    # Should return the same values
    assert default_probs == DEFAULT_PRODUCT_PROBABILITIES

    # Should be a copy, not the original
    assert default_probs is not DEFAULT_PRODUCT_PROBABILITIES

    # Modifying the returned dict shouldn't affect the original
    default_probs_copy = default_probs.copy()
    default_probs_copy["credit_card"] = 0.5
    assert DEFAULT_PRODUCT_PROBABILITIES["credit_card"] == 0.7482

def test_validate_product_probabilities_valid():
    """Test that valid probabilities pass validation."""
    valid_probs = {
        "credit_card": 0.7,
        "auto_loan": 0.2,
        "mortgage": 0.1
    }

    # Should not raise any exceptions
    validate_product_probabilities(valid_probs)

def test_validate_product_probabilities_negative():
    """Test that negative probabilities are rejected."""
    invalid_probs = {
        "credit_card": 0.7,
        "auto_loan": -0.2,
        "mortgage": 0.5
    }

    with pytest.raises(ValueError, match="negative"):
        validate_product_probabilities(invalid_probs)

def test_validate_product_probabilities_exceeds_one():
    """Test that probabilities exceeding 1 are rejected."""
    invalid_probs = {
        "credit_card": 1.1,
        "auto_loan": 0.2,
        "mortgage": 0.1
    }

    with pytest.raises(ValueError, match="exceeds 1"):
        validate_product_probabilities(invalid_probs)

def test_validate_product_probabilities_unknown_product():
    """Test that unknown product types are rejected."""
    invalid_probs = {
        "credit_card": 0.7,
        "auto_loan": 0.2,
        "unknown_product": 0.1
    }

    with pytest.raises(ValueError, match="Unknown product types"):
        validate_product_probabilities(invalid_probs)

def test_validate_product_probabilities_sum_not_one():
    """Test that probabilities not summing to 1 are rejected."""
    invalid_probs = {
        "credit_card": 0.7,
        "auto_loan": 0.2,
        "mortgage": 0.2  # Sums to 1.1
    }

    with pytest.raises(ValueError, match="sum to"):
        validate_product_probabilities(invalid_probs)

def test_assign_products_basic_functionality():
    """Test basic functionality of assign_products."""
    # Test with small sample
    assignments = assign_products(10, seed=42)

    # Should return correct number of assignments
    assert len(assignments) == 10

    # All assignments should be valid product types
    assert all(assignment in PRODUCT_TYPES for assignment in assignments)

def test_assign_products_with_default_probabilities():
    """Test assign_products with default probabilities."""
    assignments = assign_products(1000, seed=42)

    # Should only contain the three products in default probabilities
    unique_products = set(assignments)
    expected_products = set(DEFAULT_PRODUCT_PROBABILITIES.keys())
    assert unique_products.issubset(expected_products)

    # Should not contain student_loan or other_personal_loan
    assert "student_loan" not in unique_products
    assert "other_personal_loan" not in unique_products

def test_assign_products_distribution():
    """Test that generated distribution approximately matches expected probabilities."""
    n_samples = 10000
    assignments = assign_products(n_samples, seed=42)

    # Calculate actual distribution
    actual_counts = pd.Series(assignments).value_counts(normalize=True)
    actual_probs = actual_counts.to_dict()

    # Check each product's probability is close to expected
    for product, expected_prob in DEFAULT_PRODUCT_PROBABILITIES.items():
        actual_prob = actual_probs.get(product, 0.0)
        assert abs(actual_prob - expected_prob) < 0.05  # 5% tolerance

def test_assign_products_reproducibility():
    """Test that same seed produces identical results."""
    assignments1 = assign_products(100, seed=42)
    assignments2 = assign_products(100, seed=42)

    assert assignments1 == assignments2

def test_assign_products_different_seeds():
    """Test that different seeds produce different results."""
    assignments1 = assign_products(100, seed=42)
    assignments2 = assign_products(100, seed=123)

    # Results should be different (with high probability)
    assert assignments1 != assignments2

def test_assign_products_custom_probabilities():
    """Test assign_products with custom probabilities."""
    custom_probs = {
        "credit_card": 0.5,
        "auto_loan": 0.3,
        "mortgage": 0.2
    }

    assignments = assign_products(1000, probabilities=custom_probs, seed=42)

    # Calculate actual distribution
    actual_counts = pd.Series(assignments).value_counts(normalize=True)
    actual_probs = actual_counts.to_dict()

    # Check that distribution approximately matches custom probabilities
    for product, expected_prob in custom_probs.items():
        actual_prob = actual_probs.get(product, 0.0)
        assert abs(actual_prob - expected_prob) < 0.05  # 5% tolerance

def test_assign_products_include_all_products():
    """Test include_all_products option."""
    # Test with only some products specified
    partial_probs = {
        "credit_card": 0.6,
        "auto_loan": 0.4
    }

    # Should work without include_all_products=False (default)
    assignments = assign_products(100, probabilities=partial_probs, seed=42)
    unique_products = set(assignments)
    assert unique_products == {"credit_card", "auto_loan"}

    # With include_all_products=True, should include all products
    assignments_all = assign_products(100, probabilities=partial_probs,
                                    include_all_products=True, seed=42)
    unique_products_all = set(assignments_all)

    # Should include all product types
    assert unique_products_all == PRODUCT_TYPES

    # Check that probabilities sum to 1
    actual_counts = pd.Series(assignments_all).value_counts(normalize=True)
    total_prob = sum(actual_counts)
    assert np.isclose(total_prob, 1.0, atol=1e-6)

def test_assign_products_to_dataframe_basic():
    """Test basic functionality of assign_products_to_dataframe."""
    # Create test dataframe
    test_df = pd.DataFrame({
        'age': [30, 40, 25],
        'income': [5000, 6000, 4000]
    })

    # Apply product assignment
    result_df = assign_products_to_dataframe(test_df, seed=42)

    # Should have added product_type column
    assert 'product_type' in result_df.columns

    # Should have correct number of rows
    assert len(result_df) == len(test_df)

    # All assignments should be valid product types
    assert all(assignment in PRODUCT_TYPES for assignment in result_df['product_type'])

def test_assign_products_to_dataframe_custom_column():
    """Test assign_products_to_dataframe with custom column name."""
    test_df = pd.DataFrame({
        'age': [30, 40, 25],
        'income': [5000, 6000, 4000]
    })

    result_df = assign_products_to_dataframe(test_df, product_column="loan_type", seed=42)

    # Should have added custom column
    assert 'loan_type' in result_df.columns
    assert 'product_type' not in result_df.columns

def test_assign_products_to_dataframe_preserves_data():
    """Test that assign_products_to_dataframe preserves original data."""
    test_df = pd.DataFrame({
        'age': [30, 40, 25],
        'income': [5000, 6000, 4000]
    })

    original_copy = test_df.copy()
    result_df = assign_products_to_dataframe(test_df, seed=42)

    # Original columns should be unchanged
    pd.testing.assert_frame_equal(result_df[['age', 'income']], original_copy[['age', 'income']])

def test_validate_product_distribution_valid():
    """Test validate_product_distribution with valid distribution."""
    # Generate a large sample that should follow the expected distribution
    n_samples = 10000
    assignments = assign_products(n_samples, seed=42)

    # Should not raise any exceptions
    actual_probs = validate_product_distribution(
        assignments,
        DEFAULT_PRODUCT_PROBABILITIES,
        tolerance=0.05
    )

    # Should return actual probabilities
    assert isinstance(actual_probs, dict)
    assert len(actual_probs) > 0

def test_validate_product_distribution_invalid():
    """Test validate_product_distribution with invalid distribution."""
    # Create assignments that don't match expected probabilities
    invalid_assignments = ["credit_card"] * 1000  # All credit cards

    with pytest.raises(ValueError, match="deviates from expected"):
        validate_product_distribution(
            invalid_assignments,
            DEFAULT_PRODUCT_PROBABILITIES,
            tolerance=0.05
        )

def test_validate_product_distribution_small_sample():
    """Test validate_product_distribution with too small sample."""
    small_assignments = ["credit_card", "auto_loan", "mortgage"]

    with pytest.raises(ValueError, match="too small"):
        validate_product_distribution(
            small_assignments,
            DEFAULT_PRODUCT_PROBABILITIES,
            min_sample_size=100
        )

def test_set_default_product_probabilities():
    """Test set_default_product_probabilities functionality."""
    # Get original defaults
    original_defaults = get_default_product_probabilities()

    # Set new defaults
    new_probs = {
        "credit_card": 0.5,
        "auto_loan": 0.3,
        "mortgage": 0.2
    }

    set_default_product_probabilities(new_probs)

    # Check that defaults were updated
    updated_defaults = get_default_product_probabilities()
    assert updated_defaults == new_probs

    # Restore original defaults
    set_default_product_probabilities(original_defaults)

def test_set_default_product_probabilities_invalid():
    """Test that set_default_product_probabilities validates input."""
    invalid_probs = {
        "credit_card": 0.7,
        "auto_loan": 0.4  # Sum exceeds 1
    }

    with pytest.raises(ValueError):
        set_default_product_probabilities(invalid_probs)

def test_product_probabilities_exclude_student_loans():
    """Test that default probabilities correctly exclude student loans."""
    # Student loans should not be in default probabilities
    assert "student_loan" not in DEFAULT_PRODUCT_PROBABILITIES

    # Other personal loans should also not be in default probabilities
    assert "other_personal_loan" not in DEFAULT_PRODUCT_PROBABILITIES

def test_product_probabilities_correct_values():
    """Test that default probabilities have the exact specified values."""
    # Test the exact values specified in the requirements
    assert abs(DEFAULT_PRODUCT_PROBABILITIES["credit_card"] - 0.7482) < 1e-6
    assert abs(DEFAULT_PRODUCT_PROBABILITIES["auto_loan"] - 0.1280) < 1e-6
    assert abs(DEFAULT_PRODUCT_PROBABILITIES["mortgage"] - 0.1238) < 1e-6

def test_large_sample_distribution_accuracy():
    """Test distribution accuracy with very large sample."""
    n_samples = 50000
    assignments = assign_products(n_samples, seed=42)

    # Calculate actual distribution
    actual_counts = pd.Series(assignments).value_counts(normalize=True)
    actual_probs = actual_counts.to_dict()

    # With large sample, should be very close to expected
    for product, expected_prob in DEFAULT_PRODUCT_PROBABILITIES.items():
        actual_prob = actual_probs.get(product, 0.0)
        assert abs(actual_prob - expected_prob) < 0.02  # 2% tolerance for large sample

def test_assign_products_error_handling():
    """Test error handling in assign_products."""
    # Test with invalid n_borrowers
    with pytest.raises(ValueError):
        assign_products(-1)

    # Test with invalid probabilities
    with pytest.raises(ValueError):
        assign_products(10, probabilities={"invalid": 1.0})

def test_product_assignment_architecture_flexibility():
    """Test that architecture supports future additions."""
    # Test that we can add student loans and other personal loans if needed
    extended_probs = {
        "credit_card": 0.6,
        "auto_loan": 0.1,
        "mortgage": 0.1,
        "student_loan": 0.1,
        "other_personal_loan": 0.1
    }

    # This should work without breaking the architecture
    assignments = assign_products(500, probabilities=extended_probs, seed=42)

    # Should include all product types
    unique_products = set(assignments)
    assert unique_products == PRODUCT_TYPES

    # Should follow the extended distribution
    actual_counts = pd.Series(assignments).value_counts(normalize=True)
    for product, expected_prob in extended_probs.items():
        actual_prob = actual_counts.get(product, 0.0)
        assert abs(actual_prob - expected_prob) < 0.05