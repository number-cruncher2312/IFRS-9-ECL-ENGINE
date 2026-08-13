import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.balance_generator import (
    generate_balances_for_product,
    generate_balances_for_multiple_products,
    generate_ead_from_balances,
    validate_balance_generation,
    _calculate_lognormal_params,
    _generate_truncated_lognormal_samples
)
from synthetic_data.product_taxonomy import PRODUCT_TYPES, PRODUCT_TAXONOMY

def test_lognormal_parameter_calculation():
    """Test that lognormal parameters are calculated correctly."""
    # Test with known values
    min_val, median_val, max_val = 1000, 5000, 20000
    mu, sigma = _calculate_lognormal_params(min_val, median_val, max_val)

    # mu should be log(median)
    expected_mu = np.log(median_val)
    assert np.isclose(mu, expected_mu), f"mu should be log(median), got {mu} vs {expected_mu}"

    # sigma should be positive
    assert sigma > 0, f"sigma should be positive, got {sigma}"

    # Test with different ranges
    mu2, sigma2 = _calculate_lognormal_params(100, 1000, 10000)
    assert mu2 > 0 and sigma2 > 0, "Parameters should be positive for different ranges"

def test_truncated_lognormal_sample_generation():
    """Test truncated lognormal sample generation."""
    min_val, median_val, max_val = 1000, 5000, 20000
    size = 1000

    # Generate samples
    samples = _generate_truncated_lognormal_samples(min_val, median_val, max_val, size, seed=42)

    # Check basic properties
    assert len(samples) == size, f"Should generate {size} samples"
    assert np.all(samples > 0), "All samples should be positive"
    assert np.all(samples >= min_val), "All samples should be >= min_val"
    assert np.all(samples <= max_val), "All samples should be <= max_val"

    # Check that we get different results with different seeds
    samples2 = _generate_truncated_lognormal_samples(min_val, median_val, max_val, size, seed=123)
    assert not np.array_equal(samples, samples2), "Different seeds should produce different samples"

    # Check reproducibility
    samples3 = _generate_truncated_lognormal_samples(min_val, median_val, max_val, size, seed=42)
    assert np.array_equal(samples, samples3), "Same seed should produce identical samples"

def test_balance_generation_for_single_product():
    """Test balance generation for a single product."""
    product = "mortgage"
    size = 100

    # Generate balances
    balances = generate_balances_for_product(product, size, seed=42)

    # Check basic properties
    assert len(balances) == size, f"Should generate {size} balances"
    assert np.all(balances > 0), "All balances should be positive"

    # Check bounds
    config = PRODUCT_TAXONOMY[product]["balance"]
    assert np.all(balances >= config["min"]), f"Balances should be >= min ({config['min']})"
    assert np.all(balances <= config["max"]), f"Balances should be <= max ({config['max']})"

    # Check that median is reasonable
    actual_median = np.median(balances)
    expected_median = config["median"]
    # Allow 20% deviation due to sampling variability
    assert abs(actual_median - expected_median) / expected_median < 0.2, \
        f"Median should be close to expected: {actual_median} vs {expected_median}"

def test_balance_generation_for_all_products():
    """Test balance generation for all product types."""
    size = 50

    for product in PRODUCT_TYPES:
        if product in PRODUCT_TAXONOMY:  # Skip any undefined products
            balances = generate_balances_for_product(product, size, seed=42)

            # Check basic properties
            assert len(balances) == size, f"Should generate {size} balances for {product}"
            assert np.all(balances > 0), f"All balances should be positive for {product}"

            # Check bounds
            config = PRODUCT_TAXONOMY[product]["balance"]
            assert np.all(balances >= config["min"]), \
                f"Balances for {product} should be >= min ({config['min']})"
            assert np.all(balances <= config["max"]), \
                f"Balances for {product} should be <= max ({config['max']})"

def test_balance_generation_for_multiple_products():
    """Test balance generation for multiple products."""
    # Create mixed product assignments
    assignments = ["mortgage", "auto_loan", "credit_card", "mortgage", "auto_loan"]
    expected_counts = {"mortgage": 2, "auto_loan": 2, "credit_card": 1}

    # Generate balances
    balances_dict = generate_balances_for_multiple_products(assignments, seed=42)

    # Check that we got balances for all products
    assert set(balances_dict.keys()) == set(expected_counts.keys()), \
        "Should have balances for all assigned products"

    # Check counts
    for product, expected_count in expected_counts.items():
        assert len(balances_dict[product]) == expected_count, \
            f"Should have {expected_count} balances for {product}"

    # Check that all balances are valid
    for product, balances in balances_dict.items():
        config = PRODUCT_TAXONOMY[product]["balance"]
        assert np.all(balances >= config["min"]), f"Balances for {product} should respect min"
        assert np.all(balances <= config["max"]), f"Balances for {product} should respect max"
        assert np.all(balances > 0), f"All balances for {product} should be positive"

def test_ead_generation():
    """Test EAD generation (V1: EAD = balance)."""
    test_balances = np.array([1000.0, 2500.0, 5000.0, 10000.0])

    # Generate EAD
    ead_values = generate_ead_from_balances(test_balances)

    # Check that EAD equals balance
    assert np.array_equal(ead_values, test_balances), "EAD should equal balance for V1"

    # Check that it's a copy, not the same array
    assert ead_values is not test_balances, "EAD should be a copy of balances"

def test_balance_validation():
    """Test balance validation function."""
    # Create test assignments and balances
    assignments = ["mortgage", "auto_loan", "credit_card"]
    balances_dict = {
        "mortgage": np.array([200000.0, 300000.0]),
        "auto_loan": np.array([25000.0]),
        "credit_card": np.array([5000.0])
    }

    # Validate (should not raise exceptions)
    stats = validate_balance_generation(balances_dict, assignments)

    # Check that statistics are returned
    assert "mortgage" in stats, "Should have stats for mortgage"
    assert "auto_loan" in stats, "Should have stats for auto_loan"
    assert "credit_card" in stats, "Should have stats for credit_card"

    # Check that statistics contain expected fields
    for product_stats in stats.values():
        expected_fields = {"count", "min", "median", "mean", "max", "std", "skewness"}
        assert set(product_stats.keys()) == expected_fields, \
            f"Stats should contain all expected fields: {expected_fields}"

def test_balance_validation_errors():
    """Test that balance validation catches errors."""
    # Test with mismatched counts
    assignments = ["mortgage", "auto_loan"]
    balances_dict = {"mortgage": np.array([200000.0])}  # Missing auto_loan

    with pytest.raises(ValueError, match="Balances contain products not in assignments"):
        validate_balance_generation(balances_dict, assignments)

    # Test with non-positive balances
    assignments = ["mortgage"]
    balances_dict = {"mortgage": np.array([-1000.0, 200000.0])}

    with pytest.raises(ValueError, match="non-positive balances"):
        validate_balance_generation(balances_dict, assignments)

    # Test with balances below minimum
    assignments = ["mortgage"]
    config = PRODUCT_TAXONOMY["mortgage"]["balance"]
    balances_dict = {"mortgage": np.array([config["min"] - 1000, 200000.0])}

    with pytest.raises(ValueError, match="below minimum"):
        validate_balance_generation(balances_dict, assignments)

    # Test with balances above maximum
    assignments = ["mortgage"]
    balances_dict = {"mortgage": np.array([200000.0, config["max"] + 1000])}

    with pytest.raises(ValueError, match="above maximum"):
        validate_balance_generation(balances_dict, assignments)

def test_right_skewness():
    """Test that generated balances exhibit right-skewness."""
    size = 1000

    for product in ["mortgage", "auto_loan", "credit_card"]:
        balances = generate_balances_for_product(product, size, seed=42)

        # Calculate skewness
        skewness = pd.Series(balances).skew()

        # Should be positive (right-skewed)
        assert skewness > 0, f"Balances for {product} should be right-skewed (skewness: {skewness})"

        # Should be reasonably skewed (skewness > 0.5 as a threshold)
        assert skewness > 0.5, f"Balances for {product} should have meaningful right-skew (skewness: {skewness})"

def test_reproducibility():
    """Test that balance generation is reproducible with same seed."""
    size = 500
    seed = 42

    # Generate balances twice with same seed
    balances1 = generate_balances_for_product("mortgage", size, seed=seed)
    balances2 = generate_balances_for_product("mortgage", size, seed=seed)

    # Should be identical
    assert np.array_equal(balances1, balances2), "Same seed should produce identical balances"

    # Different seed should produce different results
    balances3 = generate_balances_for_product("mortgage", size, seed=123)
    assert not np.array_equal(balances1, balances3), "Different seed should produce different balances"

def test_product_differentiation():
    """Test that different products generate different balance distributions."""
    size = 1000
    seed = 42

    # Generate balances for different products
    mortgage_balances = generate_balances_for_product("mortgage", size, seed=seed)
    credit_card_balances = generate_balances_for_product("credit_card", size, seed=seed)

    # Check that means are significantly different
    mortgage_mean = np.mean(mortgage_balances)
    credit_card_mean = np.mean(credit_card_balances)

    assert mortgage_mean > credit_card_mean, \
        f"Mortgage balances should be higher than credit card: {mortgage_mean} vs {credit_card_mean}"

    # Check that the difference is substantial (at least 10x)
    ratio = mortgage_mean / credit_card_mean
    assert ratio > 10, f"Mortgage/credit card ratio should be > 10, got {ratio}"

def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test invalid product type
    with pytest.raises(ValueError, match="Unknown product_type"):
        generate_balances_for_product("invalid_product", 10, seed=42)

    # Test empty product assignments
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_balances_for_multiple_products([], seed=42)

    # Test invalid product in assignments
    with pytest.raises(ValueError, match="Invalid product types"):
        generate_balances_for_multiple_products(["invalid_product"], seed=42)

def test_large_scale_generation():
    """Test that large-scale generation works efficiently."""
    # Test with 10,000 loans (the target portfolio size)
    size = 10000
    seed = 42

    # This should complete reasonably quickly
    balances = generate_balances_for_product("mortgage", size, seed=seed)

    # Verify basic properties
    assert len(balances) == size, f"Should generate {size} balances"
    assert np.all(balances > 0), "All balances should be positive"

    # Check that we have reasonable statistical properties
    config = PRODUCT_TAXONOMY["mortgage"]["balance"]
    assert np.all(balances >= config["min"]), "All balances should respect min"
    assert np.all(balances <= config["max"]), "All balances should respect max"

    # Check that mean is reasonable
    mean_balance = np.mean(balances)
    expected_median = config["median"]
    assert 0.5 * expected_median < mean_balance < 2.0 * expected_median, \
        f"Mean should be in reasonable range: {mean_balance} vs {expected_median}"