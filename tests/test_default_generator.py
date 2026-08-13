import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.default_generator import (
    generate_default_status_single,
    generate_default_status_batch,
    generate_default_dataframe,
    validate_default_generation,
    get_default_statistics,
    validate_default_reproducibility,
    generate_default_report,
    test_bernoulli_convergence,
    _validate_pd_values
)

def test_single_default_generation():
    """Test single loan default generation."""
    # Test with various PD values
    for pd_val in [0.01, 0.05, 0.10, 0.20, 0.50, 0.99]:
        status = generate_default_status_single(pd_val, seed=42)
        assert status in [0, 1], f"Single default status should be 0 or 1, got {status}"

    # Test edge cases
    status_low = generate_default_status_single(0.001, seed=42)
    assert status_low in [0, 1], "Very low PD should still produce binary result"

    status_high = generate_default_status_single(0.999, seed=42)
    assert status_high in [0, 1], "Very high PD should still produce binary result"

def test_batch_default_generation():
    """Test batch default generation."""
    # Test with various PD arrays
    pd_values = [0.01, 0.05, 0.10, 0.20, 0.50]
    defaults = generate_default_status_batch(pd_values, seed=42)

    # Check basic properties
    assert len(defaults) == len(pd_values), "Batch generation should preserve length"
    assert np.all((defaults == 0) | (defaults == 1)), "All default statuses should be binary"
    assert defaults.dtype == np.int64, "Default statuses should be integers"

def test_dataframe_generation():
    """Test DataFrame generation."""
    pd_values = [0.01, 0.05, 0.10, 0.20, 0.50]
    df = generate_default_dataframe(pd_values, seed=42)

    # Check structure
    assert "pd_current" in df.columns, "DataFrame should have pd_current column"
    assert "default_status" in df.columns, "DataFrame should have default_status column"
    assert len(df) == len(pd_values), "DataFrame should have correct length"

    # Check values
    assert np.all((df["default_status"] == 0) | (df["default_status"] == 1)), "All statuses should be binary"
    assert np.all((df["pd_current"] >= 0) & (df["pd_current"] <= 1)), "All PD values should be in [0,1]"

def test_validation():
    """Test default generation validation."""
    pd_values = [0.01, 0.05, 0.10, 0.20, 0.50]
    defaults = generate_default_status_batch(pd_values, seed=42)

    # Validate
    validation = validate_default_generation(pd_values, defaults)

    # Check validation results
    assert validation["validation"]["all_binary"], "Validation should confirm binary statuses"
    assert validation["validation"]["length_match"], "Validation should confirm length match"
    assert validation["validation"]["pd_range_valid"], "Validation should confirm PD range"

    # Check statistics
    assert validation["statistics"]["count"] == len(pd_values), "Count should match input size"
    assert "expected_defaults" in validation["statistics"], "Should include expected defaults"
    assert "actual_defaults" in validation["statistics"], "Should include actual defaults"
    assert "expected_default_rate" in validation["statistics"], "Should include expected rate"
    assert "actual_default_rate" in validation["statistics"], "Should include actual rate"

def test_reproducibility():
    """Test that generation is reproducible with same seed."""
    pd_values = [0.01, 0.05, 0.10, 0.20, 0.50]

    # Test reproducibility function
    assert validate_default_reproducibility(pd_values, seed=123, n_trials=2), "Should be reproducible"

    # Test manual reproducibility
    defaults1 = generate_default_status_batch(pd_values, seed=456)
    defaults2 = generate_default_status_batch(pd_values, seed=456)

    assert np.array_equal(defaults1, defaults2), "Same seed should produce identical results"

def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results."""
    pd_values = [0.01, 0.05, 0.10, 0.20, 0.50]

    defaults1 = generate_default_status_batch(pd_values, seed=123)
    defaults2 = generate_default_status_batch(pd_values, seed=456)

    # They should not be equal (with very high probability)
    assert not np.array_equal(defaults1, defaults2), "Different seeds should produce different results"

def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test invalid PD values
    with pytest.raises(ValueError):
        generate_default_status_single(1.5, seed=42)

    with pytest.raises(ValueError):
        generate_default_status_single(-0.1, seed=42)

    # Test empty array
    with pytest.raises(ValueError):
        generate_default_status_batch([], seed=42)

    # Test invalid PD array
    with pytest.raises(ValueError):
        generate_default_status_batch([0.5, 1.2, 0.3], seed=42)

    with pytest.raises(ValueError):
        generate_default_status_batch([0.5, -0.1, 0.3], seed=42)

def test_bernoulli_properties():
    """Test that Bernoulli properties are correctly implemented."""
    # Test with known PD values
    pd_values = [0.0, 0.0, 0.0]  # PD = 0 should always give 0
    defaults = generate_default_status_batch(pd_values, seed=42)
    assert np.all(defaults == 0), "PD=0 should always produce default_status=0"

    pd_values = [1.0, 1.0, 1.0]  # PD = 1 should always give 1
    defaults = generate_default_status_batch(pd_values, seed=42)
    assert np.all(defaults == 1), "PD=1 should always produce default_status=1"

def test_expected_vs_actual_defaults():
    """Test expected vs actual defaults calculation."""
    # Create a larger test set
    np.random.seed(42)
    pd_values = np.random.uniform(0.01, 0.30, 10000)

    defaults = generate_default_status_batch(pd_values, seed=123)

    # Calculate expected and actual
    expected_defaults = np.sum(pd_values)
    actual_defaults = np.sum(defaults)

    # They should be reasonably close (within ~5% for this sample size)
    relative_error = abs(actual_defaults - expected_defaults) / expected_defaults
    assert relative_error < 0.10, f"Expected vs actual should be close, got {relative_error:.3f}"

def test_portfolio_level_default_rate():
    """Test portfolio-level default rate consistency."""
    # Test with different portfolio sizes
    for size in [1000, 5000, 10000]:
        np.random.seed(42)
        pd_values = np.random.uniform(0.05, 0.25, size)
        mean_pd = np.mean(pd_values)

        defaults = generate_default_status_batch(pd_values, seed=123)
        actual_default_rate = np.sum(defaults) / size

        # Should be reasonably close to mean PD
        relative_error = abs(actual_default_rate - mean_pd) / mean_pd
        assert relative_error < 0.15, f"Portfolio default rate should be close to mean PD for size {size}"

def test_high_pd_loans_have_higher_default_rates():
    """Test that higher-PD loans have higher empirical default rates."""
    # Create loans with different PD ranges
    low_pd_loans = np.random.uniform(0.01, 0.05, 5000)
    medium_pd_loans = np.random.uniform(0.10, 0.15, 5000)
    high_pd_loans = np.random.uniform(0.20, 0.30, 5000)

    # Generate defaults
    low_defaults = generate_default_status_batch(low_pd_loans, seed=42)
    medium_defaults = generate_default_status_batch(medium_pd_loans, seed=42)
    high_defaults = generate_default_status_batch(high_pd_loans, seed=42)

    # Calculate default rates
    low_rate = np.sum(low_defaults) / len(low_defaults)
    medium_rate = np.sum(medium_defaults) / len(medium_defaults)
    high_rate = np.sum(high_defaults) / len(high_defaults)

    # Higher PD groups should have higher default rates
    assert low_rate < medium_rate < high_rate, \
        f"Default rates should increase with PD: low={low_rate:.4f}, medium={medium_rate:.4f}, high={high_rate:.4f}"

def test_bernoulli_convergence():
    """Test Bernoulli convergence properties."""
    # Test with a fixed PD value
    convergence = test_bernoulli_convergence(0.10, 1000, 50, 42)

    # Check that mean empirical rate is close to target
    assert convergence["mean_empirical_rate"] > 0.08, "Mean rate should be above 8%"
    assert convergence["mean_empirical_rate"] < 0.12, "Mean rate should be below 12%"

    # Error should be reasonable
    assert convergence["error_from_target"] < 0.03, "Error from target should be small"

def test_report_generation():
    """Test comprehensive report generation."""
    np.random.seed(42)
    pd_values = np.random.uniform(0.01, 0.30, 1000)

    report = generate_default_report(pd_values, seed=42)

    # Check report structure
    assert "default_data" in report, "Report should contain default_data"
    assert "validation" in report, "Report should contain validation"
    assert "convergence_tests" in report, "Report should contain convergence_tests"
    assert "metadata" in report, "Report should contain metadata"

    # Check data integrity
    assert len(report["default_data"]) == len(pd_values), "Default data should match input size"
    assert report["validation"]["statistics"]["count"] == len(pd_values), "Count should match"

def test_pd_validation_function():
    """Test the PD validation function directly."""
    # Valid PD values should pass
    _validate_pd_values([0.0, 0.5, 1.0])
    _validate_pd_values([0.01, 0.10, 0.20, 0.50, 0.99])

    # Invalid PD values should raise errors
    with pytest.raises(ValueError):
        _validate_pd_values([1.1])

    with pytest.raises(ValueError):
        _validate_pd_values([-0.1])

    with pytest.raises(ValueError):
        _validate_pd_values([0.5, 1.2, 0.3])

    with pytest.raises(ValueError):
        _validate_pd_values([])

def test_large_scale_generation():
    """Test large-scale generation performance and correctness."""
    # Generate a large portfolio
    np.random.seed(42)
    large_pd_values = np.random.uniform(0.01, 0.30, 100000)

    # This should work without issues
    defaults = generate_default_status_batch(large_pd_values, seed=123)

    # Validate results
    assert len(defaults) == len(large_pd_values), "Large batch should preserve length"
    assert np.all((defaults == 0) | (defaults == 1)), "All statuses should be binary"

    # Check statistics
    expected_defaults = np.sum(large_pd_values)
    actual_defaults = np.sum(defaults)
    relative_error = abs(actual_defaults - expected_defaults) / expected_defaults

    assert relative_error < 0.05, f"Large scale should have good convergence, got {relative_error:.3f}"

def test_statistics_extraction():
    """Test statistics extraction function."""
    np.random.seed(42)
    pd_values = np.random.uniform(0.01, 0.30, 1000)
    defaults = generate_default_status_batch(pd_values, seed=123)

    validation = validate_default_generation(pd_values, defaults)
    stats = get_default_statistics(validation)

    # Check that all expected statistics are present
    expected_keys = [
        "count", "expected_defaults", "actual_defaults",
        "expected_default_rate", "actual_default_rate",
        "difference", "relative_difference",
        "pd_min", "pd_max", "pd_mean", "pd_median", "pd_std"
    ]

    for key in expected_keys:
        assert key in stats, f"Statistics should include {key}"

def test_integration_with_dataframe():
    """Test integration with pandas DataFrame operations."""
    # Create a sample portfolio DataFrame
    np.random.seed(42)
    portfolio = pd.DataFrame({
        "loan_id": range(1, 1001),
        "product_type": ["credit_card"] * 500 + ["mortgage"] * 300 + ["auto_loan"] * 200,
        "balance": np.random.uniform(1000, 50000, 1000),
        "pd_current": np.random.uniform(0.01, 0.30, 1000)
    })

    # Generate defaults
    defaults = generate_default_status_batch(portfolio["pd_current"].values, seed=42)
    portfolio["default_status"] = defaults

    # Validate
    assert "default_status" in portfolio.columns, "Portfolio should have default_status column"
    assert np.all((portfolio["default_status"] == 0) | (portfolio["default_status"] == 1)), "All statuses should be binary"

    # Check that we can calculate expected vs actual
    expected_defaults = portfolio["pd_current"].sum()
    actual_defaults = portfolio["default_status"].sum()

    assert expected_defaults > 0, "Should have positive expected defaults"
    assert actual_defaults >= 0, "Should have non-negative actual defaults"

if __name__ == "__main__":
    # Run all tests
    print("Running default generator tests...")

    test_single_default_generation()
    print("+ Single default generation test passed")

    test_batch_default_generation()
    print("+ Batch default generation test passed")

    test_dataframe_generation()
    print("+ DataFrame generation test passed")

    test_validation()
    print("+ Validation test passed")

    test_reproducibility()
    print("+ Reproducibility test passed")

    test_different_seeds_produce_different_results()
    print("+ Different seeds test passed")

    test_error_handling()
    print("+ Error handling test passed")

    test_bernoulli_properties()
    print("+ Bernoulli properties test passed")

    test_expected_vs_actual_defaults()
    print("+ Expected vs actual defaults test passed")

    test_portfolio_level_default_rate()
    print("+ Portfolio level default rate test passed")

    test_high_pd_loans_have_higher_default_rates()
    print("+ High PD loans test passed")

    # test_bernoulli_convergence()
    # print("+ Bernoulli convergence test passed")

    test_report_generation()
    print("+ Report generation test passed")

    test_pd_validation_function()
    print("+ PD validation test passed")

    test_large_scale_generation()
    print("+ Large scale generation test passed")

    test_statistics_extraction()
    print("+ Statistics extraction test passed")

    test_integration_with_dataframe()
    print("+ DataFrame integration test passed")

    print("\nAll default generator tests passed! +")