import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.lifetime_generator import (
    generate_lifetime_for_product,
    generate_lifetime_for_multiple_products,
    generate_lifetime_dataframe,
    validate_lifetime_values,
    get_lifetime_statistics,
    validate_lifetime_reproducibility,
    generate_lifetime_report,
    _calculate_beta_params,
    _generate_bounded_beta_samples
)
from synthetic_data.product_taxonomy import PRODUCT_TYPES, PRODUCT_TAXONOMY

def test_beta_parameter_calculation():
    """Test that beta parameters are calculated correctly."""
    # Test with known values
    min_val, median_val, max_val = 12, 48, 72
    alpha, beta = _calculate_beta_params(min_val, median_val, max_val)

    # Parameters should be positive
    assert alpha > 0, f"alpha should be positive, got {alpha}"
    assert beta > 0, f"beta should be positive, got {beta}"

    # Test with different ranges
    alpha2, beta2 = _calculate_beta_params(60, 180, 360)
    assert alpha2 > 0 and beta2 > 0, "Parameters should be positive for different ranges"

def test_bounded_beta_sample_generation():
    """Test bounded beta sample generation."""
    min_val, median_val, max_val = 12, 48, 72
    size = 1000

    # Generate samples
    samples = _generate_bounded_beta_samples(min_val, median_val, max_val, size, seed=42)

    # Check basic properties
    assert len(samples) == size, f"Should generate {size} samples"
    assert np.all(samples > 0), "All samples should be positive"
    assert np.all(samples >= min_val), "All samples should be >= min_val"
    assert np.all(samples <= max_val), "All samples should be <= max_val"
    assert np.all(samples == np.round(samples)), "All samples should be integers"

def test_single_product_lifetime_generation():
    """Test lifetime generation for single products."""
    for product in ["mortgage", "auto_loan", "credit_card"]:
        # Test with different sizes
        for size in [1, 10, 100]:
            lifetimes = generate_lifetime_for_product(product, size, seed=42)

            # Check basic properties
            assert len(lifetimes) == size, f"Should generate {size} lifetimes for {product}"
            assert np.all(lifetimes > 0), f"{product} lifetimes should be positive"
            assert np.all(lifetimes == np.round(lifetimes)), f"{product} lifetimes should be integers"

            # Check bounds
            config = PRODUCT_TAXONOMY[product]["remaining_lifetime_months"]
            assert np.all(lifetimes >= config["min"]), f"{product} lifetimes should respect min"
            assert np.all(lifetimes <= config["max"]), f"{product} lifetimes should respect max"

def test_multiple_products_lifetime_generation():
    """Test lifetime generation for multiple products."""
    # Test with mixed product assignments
    assignments = ["mortgage", "auto_loan", "credit_card", "mortgage", "auto_loan"]

    lifetimes_dict = generate_lifetime_for_multiple_products(assignments, seed=42)

    # Check structure
    assert set(lifetimes_dict.keys()) == {"mortgage", "auto_loan", "credit_card"}
    assert len(lifetimes_dict["mortgage"]) == 2
    assert len(lifetimes_dict["auto_loan"]) == 2
    assert len(lifetimes_dict["credit_card"]) == 1

    # Check values for each product
    for product, lifetimes in lifetimes_dict.items():
        config = PRODUCT_TAXONOMY[product]["remaining_lifetime_months"]
        assert np.all(lifetimes >= config["min"]), f"{product} lifetimes should respect min"
        assert np.all(lifetimes <= config["max"]), f"{product} lifetimes should respect max"
        assert np.all(lifetimes > 0), f"{product} lifetimes should be positive"
        assert np.all(lifetimes == np.round(lifetimes)), f"{product} lifetimes should be integers"

def test_dataframe_generation():
    """Test DataFrame generation."""
    assignments = ["mortgage", "auto_loan", "credit_card", "mortgage"]

    lifetime_df = generate_lifetime_dataframe(assignments, seed=42)

    # Check structure
    assert len(lifetime_df) == len(assignments), "DataFrame should have same length as assignments"
    assert "product_type" in lifetime_df.columns, "Should have product_type column"
    assert "remaining_lifetime_months" in lifetime_df.columns, "Should have remaining_lifetime_months column"

    # Check values
    for product in lifetime_df["product_type"].unique():
        product_data = lifetime_df[lifetime_df["product_type"] == product]
        product_lifetimes = product_data["remaining_lifetime_months"].values

        config = PRODUCT_TAXONOMY[product]["remaining_lifetime_months"]
        assert np.all(product_lifetimes >= config["min"]), f"{product} lifetimes should respect min"
        assert np.all(product_lifetimes <= config["max"]), f"{product} lifetimes should respect max"

def test_lifetime_validation():
    """Test lifetime value validation."""
    # Generate test data
    lifetimes = generate_lifetime_for_product("auto_loan", 50, seed=42)

    # Validate
    validation = validate_lifetime_values(lifetimes, "auto_loan")

    # Check validation results
    assert validation["all_within_bounds"], "All lifetimes should be within bounds"
    assert validation["all_positive"], "All lifetimes should be positive"
    assert validation["all_integers"], "All lifetimes should be integers"
    assert validation["reasonably_centered"], "Lifetimes should be reasonably centered around median"

def test_lifetime_statistics():
    """Test lifetime statistics calculation."""
    # Generate test data
    lifetimes = generate_lifetime_for_product("mortgage", 100, seed=42)

    # Calculate statistics
    stats = get_lifetime_statistics(lifetimes, "mortgage")

    # Check statistics
    assert stats["count"] == 100, "Count should match input size"
    assert "min" in stats, "Statistics should include min"
    assert "max" in stats, "Statistics should include max"
    assert "mean" in stats, "Statistics should include mean"
    assert "median" in stats, "Statistics should include median"
    assert "std" in stats, "Statistics should include std"
    assert "median_config" in stats, "Statistics should include median_config"

    # Check that statistics are reasonable
    config = PRODUCT_TAXONOMY["mortgage"]["remaining_lifetime_months"]
    assert stats["min"] >= config["min"], "Min statistic should respect config min"
    assert stats["max"] <= config["max"], "Max statistic should respect config max"

def test_reproducibility():
    """Test that generation is reproducible with same seed."""
    assignments = ["mortgage", "auto_loan", "credit_card", "mortgage"]

    # Test reproducibility function
    assert validate_lifetime_reproducibility(assignments, seed=123, n_trials=2), "Should be reproducible"

    # Test manual reproducibility
    df1 = generate_lifetime_dataframe(assignments, seed=456)
    df2 = generate_lifetime_dataframe(assignments, seed=456)

    assert df1.equals(df2), "DataFrames with same seed should be identical"

def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results."""
    assignments = ["mortgage", "auto_loan", "credit_card"]

    df1 = generate_lifetime_dataframe(assignments, seed=123)
    df2 = generate_lifetime_dataframe(assignments, seed=456)

    # They should not be equal
    assert not df1.equals(df2), "Different seeds should produce different results"

def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test invalid product type
    with pytest.raises(ValueError):
        generate_lifetime_for_product("invalid_product", 10)

    # Test invalid size
    with pytest.raises(ValueError):
        generate_lifetime_for_product("mortgage", 0)

    # Test empty assignments
    with pytest.raises(ValueError):
        generate_lifetime_for_multiple_products([])

    # Test invalid product in assignments
    with pytest.raises(ValueError):
        generate_lifetime_for_multiple_products(["mortgage", "invalid_product"])

def test_large_scale_generation():
    """Test large-scale generation."""
    # Test with large number of loans
    large_assignments = ["mortgage"] * 10000 + ["auto_loan"] * 5000 + ["credit_card"] * 2000

    lifetimes_dict = generate_lifetime_for_multiple_products(large_assignments, seed=42)

    # Check counts
    assert len(lifetimes_dict["mortgage"]) == 10000
    assert len(lifetimes_dict["auto_loan"]) == 5000
    assert len(lifetimes_dict["credit_card"]) == 2000

    # Check all values are valid
    for product, lifetimes in lifetimes_dict.items():
        config = PRODUCT_TAXONOMY[product]["remaining_lifetime_months"]
        assert np.all(lifetimes >= config["min"]), f"{product} lifetimes should respect min"
        assert np.all(lifetimes <= config["max"]), f"{product} lifetimes should respect max"
        assert np.all(lifetimes > 0), f"{product} lifetimes should be positive"

def test_report_generation():
    """Test comprehensive report generation."""
    assignments = ["mortgage", "auto_loan", "credit_card", "mortgage", "auto_loan"]

    report = generate_lifetime_report(assignments, seed=42)

    # Check report structure
    assert "lifetime_data" in report, "Report should contain lifetime_data"
    assert "statistics" in report, "Report should contain statistics"
    assert "validation" in report, "Report should contain validation"
    assert "overall_validation" in report, "Report should contain overall_validation"
    assert "metadata" in report, "Report should contain metadata"

    # Check data integrity
    assert len(report["lifetime_data"]) == len(assignments), "Lifetime data should match assignment count"
    assert len(report["statistics"]) == 3, "Should have statistics for 3 products"
    assert report["overall_validation"]["all_products_valid"], "All products should be valid"

def test_product_differentiation():
    """Test that different products generate different lifetime distributions."""
    # Generate lifetimes for different products with same seed
    mortgage_lifetimes = generate_lifetime_for_product("mortgage", 1000, seed=42)
    auto_lifetimes = generate_lifetime_for_product("auto_loan", 1000, seed=42)
    credit_lifetimes = generate_lifetime_for_product("credit_card", 1000, seed=42)

    # Calculate means
    mortgage_mean = np.mean(mortgage_lifetimes)
    auto_mean = np.mean(auto_lifetimes)
    credit_mean = np.mean(credit_lifetimes)

    # Check that means are different (product differentiation)
    assert mortgage_mean != auto_mean, "Mortgage and auto loan means should differ"
    assert mortgage_mean != credit_mean, "Mortgage and credit card means should differ"
    assert auto_mean != credit_mean, "Auto loan and credit card means should differ"

    # Check that means are reasonably close to configured medians
    mortgage_config_median = PRODUCT_TAXONOMY["mortgage"]["remaining_lifetime_months"]["median"]
    auto_config_median = PRODUCT_TAXONOMY["auto_loan"]["remaining_lifetime_months"]["median"]
    credit_config_median = PRODUCT_TAXONOMY["credit_card"]["remaining_lifetime_months"]["median"]

    assert abs(mortgage_mean - mortgage_config_median) < mortgage_config_median * 0.3, \
        "Mortgage mean should be reasonably close to configured median"
    assert abs(auto_mean - auto_config_median) < auto_config_median * 0.3, \
        "Auto loan mean should be reasonably close to configured median"
    assert abs(credit_mean - credit_config_median) < credit_config_median * 0.3, \
        "Credit card mean should be reasonably close to configured median"

def test_inactive_products():
    """Test that inactive V1 products still work but aren't required."""
    # These products are defined in taxonomy but inactive for V1
    # They should still work if called directly
    for product in ["student_loan", "other_personal_loan"]:
        lifetimes = generate_lifetime_for_product(product, 10, seed=42)

        assert len(lifetimes) == 10, f"Should generate lifetimes for {product}"
        config = PRODUCT_TAXONOMY[product]["remaining_lifetime_months"]
        assert np.all(lifetimes >= config["min"]), f"{product} lifetimes should respect min"
        assert np.all(lifetimes <= config["max"]), f"{product} lifetimes should respect max"

if __name__ == "__main__":
    # Run all tests
    print("Running lifetime generator tests...")

    test_beta_parameter_calculation()
    print("+ Beta parameter calculation test passed")

    test_bounded_beta_sample_generation()
    print("+ Bounded beta sample generation test passed")

    test_single_product_lifetime_generation()
    print("+ Single product lifetime generation test passed")

    test_multiple_products_lifetime_generation()
    print("+ Multiple products lifetime generation test passed")

    test_dataframe_generation()
    print("+ DataFrame generation test passed")

    test_lifetime_validation()
    print("+ Lifetime validation test passed")

    test_lifetime_statistics()
    print("+ Lifetime statistics test passed")

    test_reproducibility()
    print("+ Reproducibility test passed")

    test_different_seeds_produce_different_results()
    print("+ Different seeds test passed")

    test_error_handling()
    print("+ Error handling test passed")

    test_large_scale_generation()
    print("+ Large scale generation test passed")

    test_report_generation()
    print("+ Report generation test passed")

    test_product_differentiation()
    print("+ Product differentiation test passed")

    test_inactive_products()
    print("+ Inactive products test passed")

    print("\nAll lifetime generator tests passed! +")
