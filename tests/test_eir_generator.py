import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.eir_generator import (
    generate_eir_for_product,
    generate_eir_for_multiple_products,
    generate_eir_dataframe,
    validate_eir_values,
    get_eir_statistics,
    validate_eir_reproducibility,
    generate_eir_report,
    _run_validation_tests
)
from synthetic_data.product_taxonomy import PRODUCT_TYPES

def test_single_product_eir_generation():
    """Test EIR generation for a single product."""
    # Test mortgage EIR
    mortgage_eir = generate_eir_for_product("mortgage", n_loans=1, seed=42)
    assert isinstance(mortgage_eir, float), "Single EIR should be float"
    assert 0.045 <= mortgage_eir <= 0.075, f"Mortgage EIR {mortgage_eir} out of bounds"

    # Test auto loan EIR
    auto_eir = generate_eir_for_product("auto_loan", n_loans=1, seed=42)
    assert isinstance(auto_eir, float), "Single EIR should be float"
    assert 0.04 <= auto_eir <= 0.09, f"Auto loan EIR {auto_eir} out of bounds"

    # Test credit card EIR
    credit_card_eir = generate_eir_for_product("credit_card", n_loans=1, seed=42)
    assert isinstance(credit_card_eir, float), "Single EIR should be float"
    assert 0.12 <= credit_card_eir <= 0.25, f"Credit card EIR {credit_card_eir} out of bounds"

def test_multiple_eir_values():
    """Test generation of multiple EIR values for same product."""
    auto_eirs = generate_eir_for_product("auto_loan", n_loans=10, seed=42)
    assert isinstance(auto_eirs, list), "Multiple EIRs should be list"
    assert len(auto_eirs) == 10, "Should generate correct number of EIRs"
    assert all(0.04 <= eir <= 0.09 for eir in auto_eirs), "Auto EIRs should be within bounds"

def test_multiple_products():
    """Test EIR generation for multiple different products."""
    product_assignments = ["credit_card", "mortgage", "auto_loan", "credit_card"]
    eir_dict = generate_eir_for_multiple_products(product_assignments, seed=42)

    assert set(eir_dict.keys()) == {"credit_card", "mortgage", "auto_loan"}, "Should have all products"
    assert len(eir_dict["credit_card"]) == 2, "Credit card should have 2 EIRs"
    assert len(eir_dict["mortgage"]) == 1, "Mortgage should have 1 EIR"
    assert len(eir_dict["auto_loan"]) == 1, "Auto loan should have 1 EIR"

def test_dataframe_generation():
    """Test EIR DataFrame generation."""
    product_assignments = ["credit_card", "mortgage", "auto_loan", "credit_card"]
    eir_df = generate_eir_dataframe(product_assignments, seed=42)

    assert len(eir_df) == len(product_assignments), "DataFrame should have same length as assignments"
    assert "product_type" in eir_df.columns, "Should have product_type column"
    assert "eir" in eir_df.columns, "Should have eir column"

    # Check that product types match
    assert list(eir_df["product_type"]) == product_assignments, "Product types should match assignments"

def test_eir_validation():
    """Test EIR value validation."""
    auto_eirs = generate_eir_for_product("auto_loan", n_loans=10, seed=42)
    validation = validate_eir_values(auto_eirs, "auto_loan")

    assert validation["all_within_bounds"], "All EIRs should be within bounds"
    assert validation["all_positive"], "All EIRs should be positive"
    assert validation["has_variation"], "Should have variation for multiple values"

def test_eir_statistics():
    """Test EIR statistics calculation."""
    auto_eirs = generate_eir_for_product("auto_loan", n_loans=10, seed=42)
    stats = get_eir_statistics(auto_eirs, "auto_loan")

    expected_fields = {"count", "min", "median", "mean", "max", "std", "base_eir", "distance_from_base", "coefficient_of_variation"}
    assert set(stats.keys()) == expected_fields, f"Statistics should contain all expected fields: {expected_fields}"

    assert stats["count"] == 10, "Count should match number of EIRs"
    assert stats["min"] >= 0.04, "Min should be at least product min"
    assert stats["max"] <= 0.09, "Max should be at most product max"
    assert stats["base_eir"] == 0.065, "Base EIR should match product configuration"

def test_eir_reproducibility():
    """Test EIR generation reproducibility."""
    product_assignments = ["mortgage", "auto_loan", "credit_card"]
    result = validate_eir_reproducibility(product_assignments, seed=123, n_trials=2)
    assert result is True, "EIR generation should be reproducible with same seed"

def test_all_products_eir_generation():
    """Test EIR generation for all product types."""
    products = ["mortgage", "auto_loan", "credit_card", "student_loan", "other_personal_loan"]
    expected_ranges = {
        "mortgage": (0.045, 0.075),
        "auto_loan": (0.04, 0.09),
        "credit_card": (0.12, 0.25),
        "student_loan": (0.04, 0.08),
        "other_personal_loan": (0.08, 0.22)
    }

    for product in products:
        eirs = generate_eir_for_product(product, n_loans=10, seed=42)
        min_eir, max_eir = expected_ranges[product]

        # Check bounds
        assert all(min_eir <= eir <= max_eir for eir in eirs), f"{product} EIRs out of bounds"

        # Check statistics
        stats = get_eir_statistics(eirs, product)
        assert stats["count"] == 10, f"{product} should have 10 EIR values"

def test_eir_report_generation():
    """Test comprehensive EIR report generation."""
    product_assignments = ["credit_card", "mortgage", "auto_loan", "credit_card"]
    report = generate_eir_report(product_assignments, seed=42)

    expected_keys = {"eir_data", "statistics", "validation", "overall_validation", "metadata"}
    assert set(report.keys()) == expected_keys, f"Report should contain all expected keys: {expected_keys}"

    # Check EIR data
    eir_data = report["eir_data"]
    assert len(eir_data) == len(product_assignments), "EIR data should match assignments"

    # Check statistics
    statistics = report["statistics"]
    assert len(statistics) > 0, "Should have statistics for at least one product"

    # Check validation
    validation = report["validation"]
    assert len(validation) > 0, "Should have validation for at least one product"

    # Check overall validation
    overall_validation = report["overall_validation"]
    assert overall_validation["all_products_valid"], "All products should be valid"

    # Check metadata
    metadata = report["metadata"]
    assert metadata["seed"] == 42, "Seed should match"
    assert metadata["total_loans"] == len(product_assignments), "Total loans should match"

def test_eir_bounds_compliance():
    """Test that EIR values always comply with product bounds."""
    products = ["mortgage", "auto_loan", "credit_card", "student_loan", "other_personal_loan"]

    for product in products:
        # Generate many EIR values to test bounds compliance
        eirs = generate_eir_for_product(product, n_loans=100, seed=42)

        # Get expected bounds from product taxonomy
        from synthetic_data.product_taxonomy import PRODUCT_TAXONOMY
        eir_config = PRODUCT_TAXONOMY[product]["eir"]
        min_eir = eir_config["min"]
        max_eir = eir_config["max"]

        # Check all values are within bounds
        assert all(min_eir <= eir <= max_eir for eir in eirs), f"{product} EIRs violate bounds"

def test_eir_centers_around_base():
    """Test that EIR values center around the configured base rates."""
    products = ["mortgage", "auto_loan", "credit_card"]
    n_samples = 1000

    for product in products:
        eirs = generate_eir_for_product(product, n_loans=n_samples, seed=42)
        stats = get_eir_statistics(eirs, product)

        # Get expected bounds
        from synthetic_data.product_taxonomy import PRODUCT_TAXONOMY
        eir_config = PRODUCT_TAXONOMY[product]["eir"]
        min_eir = eir_config["min"]
        max_eir = eir_config["max"]
        base_eir = eir_config["base"]

        # Check that mean is close to base (within 10% of range)
        eir_range = max_eir - min_eir
        tolerance = eir_range * 0.1
        distance_from_base = abs(stats['mean'] - base_eir)

        assert distance_from_base <= tolerance, f"{product} mean {stats['mean']} too far from base {base_eir}"

def test_eir_error_handling():
    """Test EIR generator error handling."""
    # Test invalid product type
    with pytest.raises(ValueError, match="Unknown product_type"):
        generate_eir_for_product("invalid_product", n_loans=1)

    # Test invalid n_loans
    with pytest.raises(ValueError, match="n_loans must be positive"):
        generate_eir_for_product("mortgage", n_loans=0)

    # Test invalid variation scale
    with pytest.raises(ValueError, match="variation_scale must be in"):
        generate_eir_for_product("mortgage", n_loans=1, variation_scale=1.5)

    # Test empty product assignments
    with pytest.raises(ValueError, match="product_assignments cannot be empty"):
        generate_eir_for_multiple_products([])

    # Test invalid product in assignments
    with pytest.raises(ValueError, match="Invalid product types"):
        generate_eir_for_multiple_products(["mortgage", "invalid_product"])

def test_eir_product_differentiation():
    """Test that different products have different EIR characteristics."""
    # Generate EIRs for different products
    mortgage_eirs = generate_eir_for_product("mortgage", n_loans=100, seed=42)
    credit_card_eirs = generate_eir_for_product("credit_card", n_loans=100, seed=42)

    # Calculate statistics
    mortgage_stats = get_eir_statistics(mortgage_eirs, "mortgage")
    credit_card_stats = get_eir_statistics(credit_card_eirs, "credit_card")

    # Mortgage should have lower EIRs than credit cards
    assert mortgage_stats["mean"] < credit_card_stats["mean"], "Mortgage EIRs should be lower than credit card EIRs"
    assert mortgage_stats["max"] < credit_card_stats["min"], "Mortgage max should be less than credit card min"

def test_eir_variation_scale():
    """Test that variation scale parameter works correctly."""
    # Test with low variation
    low_var_eirs = generate_eir_for_product("auto_loan", n_loans=100, seed=42, variation_scale=0.1)
    low_var_stats = get_eir_statistics(low_var_eirs, "auto_loan")

    # Test with high variation
    high_var_eirs = generate_eir_for_product("auto_loan", n_loans=100, seed=42, variation_scale=0.9)
    high_var_stats = get_eir_statistics(high_var_eirs, "auto_loan")

    # High variation should have higher standard deviation
    assert high_var_stats["std"] > low_var_stats["std"], "Higher variation scale should produce higher standard deviation"

def test_internal_validation_tests():
    """Test that internal validation tests pass."""
    # This should not raise any exceptions
    _run_validation_tests()

def test_eir_continuous_values():
    """Test that EIR values are continuous (not discrete)."""
    # Generate many EIR values
    eirs = generate_eir_for_product("auto_loan", n_loans=1000, seed=42)

    # Count unique values
    unique_eirs = set(eirs)
    unique_count = len(unique_eirs)

    # Should have many unique values (continuous distribution)
    assert unique_count > 100, f"Should have many unique EIR values, got {unique_count}"

def test_eir_no_current_future_info():
    """Test that EIR generation doesn't use current/future information."""
    # This is a conceptual test - the EIR generator should only use:
    # - product_type
    # - base_eir from taxonomy
    # - stochastic variation
    # - seed for reproducibility

    # The generator should NOT use:
    # - PD_current
    # - current borrower characteristics
    # - default status
    # - staging
    # - LGD
    # - EAD
    # - any other future/current outcome

    # We can verify this by checking the function signatures and implementation
    import inspect

    # Check generate_eir_for_product signature
    sig = inspect.signature(generate_eir_for_product)
    params = list(sig.parameters.keys())
    expected_params = ['product_type', 'n_loans', 'seed', 'variation_scale']
    assert set(params) == set(expected_params), f"Function should only accept {expected_params}"

    # Check that the function doesn't reference forbidden concepts
    source = inspect.getsource(generate_eir_for_product)
    forbidden_concepts = ['PD_current', 'default', 'staging', 'LGD', 'EAD', 'current_', 'future']
    for concept in forbidden_concepts:
        assert concept not in source, f"EIR generator should not reference {concept}"

def test_eir_seed_reproducibility_detailed():
    """Test detailed seed reproducibility."""
    product_assignments = ["mortgage", "auto_loan", "credit_card", "student_loan"]

    # Generate with same seed multiple times
    eir_df1 = generate_eir_dataframe(product_assignments, seed=999)
    eir_df2 = generate_eir_dataframe(product_assignments, seed=999)

    # Should be identical
    assert eir_df1.equals(eir_df2), "Same seed should produce identical results"

    # Different seed should produce different results
    eir_df3 = generate_eir_dataframe(product_assignments, seed=1000)
    assert not eir_df1.equals(eir_df3), "Different seeds should produce different results"

def test_eir_large_scale_generation():
    """Test EIR generation at scale (like full portfolio)."""
    # Generate EIRs for a large portfolio
    n_loans = 10000
    product_assignments = ["credit_card"] * 7500 + ["auto_loan"] * 1300 + ["mortgage"] * 1200

    # This should work without issues
    eir_df = generate_eir_dataframe(product_assignments, seed=42)

    assert len(eir_df) == n_loans, "Should generate correct number of EIRs"
    assert eir_df["product_type"].value_counts()["credit_card"] == 7500, "Credit card count should match"
    assert eir_df["product_type"].value_counts()["auto_loan"] == 1300, "Auto loan count should match"
    assert eir_df["product_type"].value_counts()["mortgage"] == 1200, "Mortgage count should match"

def test_eir_statistical_report():
    """Test generation of comprehensive statistical report."""
    # Generate a report for a mixed portfolio
    product_assignments = ["credit_card"] * 100 + ["auto_loan"] * 50 + ["mortgage"] * 30 + ["student_loan"] * 20
    report = generate_eir_report(product_assignments, seed=42)

    # Verify report structure
    assert "statistics" in report, "Report should contain statistics"
    assert "validation" in report, "Report should contain validation"
    assert "metadata" in report, "Report should contain metadata"

    # Check statistics for each product
    statistics = report["statistics"]
    for product in ["credit_card", "auto_loan", "mortgage", "student_loan"]:
        assert product in statistics, f"Report should contain statistics for {product}"
        product_stats = statistics[product]

        # Verify key statistical measures
        assert "mean" in product_stats, "Should have mean"
        assert "std" in product_stats, "Should have standard deviation"
        assert "base_eir" in product_stats, "Should have base EIR"
        assert "distance_from_base" in product_stats, "Should have distance from base"

        # Verify values are reasonable
        assert product_stats["std"] > 0, "Standard deviation should be positive"
        assert product_stats["count"] > 0, "Count should be positive"

    print("EIR statistical report test passed!")