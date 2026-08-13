import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.portfolio_generator import (
    generate_base_portfolio,
    generate_portfolio_with_statistics,
    validate_portfolio_reproducibility,
    validate_portfolio_properties
)
from synthetic_data.product_assignment import DEFAULT_PRODUCT_PROBABILITIES
from synthetic_data.product_taxonomy import PRODUCT_TYPES

def test_basic_portfolio_generation():
    """Test basic portfolio generation functionality."""
    # Generate small portfolio for testing
    portfolio = generate_base_portfolio(n_loans=100, seed=42)

    # Check basic properties
    assert len(portfolio) == 100, "Should generate 100 loans"
    assert "loan_id" in portfolio.columns, "Should have loan_id column"
    assert "product_type" in portfolio.columns, "Should have product_type column"
    assert "balance" in portfolio.columns, "Should have balance column"
    assert "ead" in portfolio.columns, "Should have ead column"

    # Check that loan_ids are unique and sequential
    assert portfolio["loan_id"].is_unique, "Loan IDs should be unique"
    assert (portfolio["loan_id"] == range(1, 101)).all(), "Loan IDs should be sequential from 1"

def test_portfolio_with_default_products():
    """Test that portfolio uses V1 active products by default."""
    portfolio = generate_base_portfolio(n_loans=1000, seed=42)

    # Check that only V1 active products are used
    unique_products = set(portfolio["product_type"].unique())
    expected_products = set(DEFAULT_PRODUCT_PROBABILITIES.keys())

    assert unique_products.issubset(expected_products), \
        f"Should only use V1 active products: {unique_products} vs {expected_products}"

    # Should not include student loans or other personal loans by default
    assert "student_loan" not in unique_products, "Should not include student loans by default"
    assert "other_personal_loan" not in unique_products, "Should not include other personal loans by default"

def test_ead_equals_balance():
    """Test that EAD equals balance for all loans (V1 specification)."""
    portfolio = generate_base_portfolio(n_loans=500, seed=42)

    # Check that EAD equals balance for all rows
    assert np.allclose(portfolio["balance"], portfolio["ead"]), "EAD should equal balance for all loans"

    # Check that they are separate columns (not the same object)
    assert portfolio["balance"] is not portfolio["ead"], "Balance and EAD should be separate columns"

def test_portfolio_reproducibility():
    """Test that portfolio generation is reproducible with same seed."""
    # Generate portfolios with same seed
    portfolio1 = generate_base_portfolio(n_loans=200, seed=123)
    portfolio2 = generate_base_portfolio(n_loans=200, seed=123)

    # Should be identical
    assert portfolio1.equals(portfolio2), "Same seed should produce identical portfolios"

    # Different seed should produce different results
    portfolio3 = generate_base_portfolio(n_loans=200, seed=456)
    assert not portfolio1.equals(portfolio3), "Different seeds should produce different portfolios"

def test_portfolio_reproducibility_function():
    """Test the portfolio reproducibility validation function."""
    # Should return True for reproducible generation
    result = validate_portfolio_reproducibility(n_loans=100, seed=789, n_trials=3)
    assert result is True, "Portfolio generation should be reproducible"

def test_portfolio_with_statistics():
    """Test portfolio generation with statistics."""
    result = generate_portfolio_with_statistics(n_loans=500, seed=42)

    # Check that all expected components are returned
    assert "portfolio" in result, "Should return portfolio DataFrame"
    assert "statistics" in result, "Should return statistics"
    assert "metadata" in result, "Should return metadata"

    # Check portfolio properties
    portfolio = result["portfolio"]
    assert len(portfolio) == 500, "Portfolio should have correct size"
    assert "loan_id" in portfolio.columns, "Portfolio should have required columns"

    # Check statistics
    statistics = result["statistics"]
    assert len(statistics) > 0, "Should have statistics for at least one product"

    # Check that statistics contain expected fields
    for product, stats in statistics.items():
        expected_fields = {"count", "min", "median", "mean", "max", "std", "skewness", "sum"}
        assert set(stats.keys()) == expected_fields, \
            f"Statistics should contain all expected fields: {expected_fields}"

    # Check metadata
    metadata = result["metadata"]
    assert metadata["n_loans"] == 500, "Metadata should contain correct n_loans"
    assert metadata["seed"] == 42, "Metadata should contain correct seed"
    assert "generation_timestamp" in metadata, "Metadata should contain timestamp"
    assert "product_distribution" in metadata, "Metadata should contain product distribution"

def test_portfolio_validation():
    """Test portfolio validation function."""
    portfolio = generate_base_portfolio(n_loans=100, seed=42)

    # Validate portfolio
    validation = validate_portfolio_properties(portfolio)

    # Check validation results
    assert validation["has_required_columns"] is True, "Should have required columns"
    assert validation["loan_ids_unique"] is True, "Loan IDs should be unique"
    assert validation["loan_ids_sequential"] is True, "Loan IDs should be sequential"
    assert validation["all_balances_positive"] is True, "All balances should be positive"
    assert validation["ead_equals_balance"] is True, "EAD should equal balance"

    # Check right-skewness by product
    skewness_results = validation["right_skewness_by_product"]
    for product, is_right_skewed in skewness_results.items():
        assert is_right_skewed is True, f"Balances for {product} should be right-skewed"

def test_portfolio_size_validation():
    """Test that portfolio generation validates size parameter."""
    # Test with zero loans
    with pytest.raises(ValueError, match="must be positive"):
        generate_base_portfolio(n_loans=0, seed=42)

    # Test with negative loans
    with pytest.raises(ValueError, match="must be positive"):
        generate_base_portfolio(n_loans=-100, seed=42)

def test_large_portfolio_generation():
    """Test generation of the target 10,000-loan portfolio."""
    # Generate the full 10,000-loan portfolio
    portfolio = generate_base_portfolio(n_loans=10000, seed=42)

    # Check basic properties
    assert len(portfolio) == 10000, "Should generate exactly 10,000 loans"
    assert portfolio["loan_id"].is_unique, "All loan IDs should be unique"
    assert (portfolio["loan_id"] == range(1, 10001)).all(), "Loan IDs should be sequential"

    # Check that all balances are positive
    assert (portfolio["balance"] > 0).all(), "All balances should be positive"

    # Check that EAD equals balance
    assert np.allclose(portfolio["balance"], portfolio["ead"]), "EAD should equal balance"

    # Check product distribution
    product_counts = portfolio["product_type"].value_counts(normalize=True)
    for product, expected_prob in DEFAULT_PRODUCT_PROBABILITIES.items():
        actual_prob = product_counts.get(product, 0.0)
        # Allow 5% tolerance for sampling variability
        assert abs(actual_prob - expected_prob) < 0.05, \
            f"Product {product} distribution should be close to expected: {actual_prob} vs {expected_prob}"

def test_portfolio_statistics_accuracy():
    """Test that portfolio statistics are calculated correctly."""
    # Generate portfolio with statistics
    result = generate_portfolio_with_statistics(n_loans=1000, seed=42)
    portfolio = result["portfolio"]
    statistics = result["statistics"]

    # Verify statistics for each product
    for product in portfolio["product_type"].unique():
        product_data = portfolio[portfolio["product_type"] == product]
        product_stats = statistics[product]

        # Check count
        assert product_stats["count"] == len(product_data), \
            f"Count should match for {product}"

        # Check min, max
        assert product_stats["min"] == float(product_data["balance"].min()), \
            f"Min should match for {product}"
        assert product_stats["max"] == float(product_data["balance"].max()), \
            f"Max should match for {product}"

        # Check mean (allow small floating point differences)
        assert abs(product_stats["mean"] - float(product_data["balance"].mean())) < 1e-6, \
            f"Mean should match for {product}"

        # Check sum
        assert abs(product_stats["sum"] - float(product_data["balance"].sum())) < 1e-6, \
            f"Sum should match for {product}"

def test_include_all_products_option():
    """Test the include_all_products option."""
    # Generate portfolio with all products
    portfolio_all = generate_base_portfolio(n_loans=1000, seed=42, include_all_products=True)

    # Should include all product types
    unique_products = set(portfolio_all["product_type"].unique())
    assert unique_products == PRODUCT_TYPES, \
        f"Should include all products when include_all_products=True: {unique_products} vs {PRODUCT_TYPES}"

    # Generate portfolio with default (V1 active only)
    portfolio_default = generate_base_portfolio(n_loans=1000, seed=42, include_all_products=False)

    # Should only include V1 active products
    unique_products_default = set(portfolio_default["product_type"].unique())
    expected_default = set(DEFAULT_PRODUCT_PROBABILITIES.keys())
    assert unique_products_default == expected_default, \
        f"Should only include V1 active products by default: {unique_products_default} vs {expected_default}"

def test_portfolio_product_differentiation():
    """Test that different products have different balance characteristics."""
    portfolio = generate_base_portfolio(n_loans=2000, seed=42)

    # Get statistics by product
    stats_by_product = {}
    for product in portfolio["product_type"].unique():
        product_data = portfolio[portfolio["product_type"] == product]
        stats_by_product[product] = {
            "mean_balance": product_data["balance"].mean(),
            "median_balance": product_data["balance"].median(),
            "max_balance": product_data["balance"].max()
        }

    # Check that mortgage balances are higher than credit card balances
    mortgage_stats = stats_by_product.get("mortgage", {})
    credit_card_stats = stats_by_product.get("credit_card", {})

    if mortgage_stats and credit_card_stats:
        assert mortgage_stats["mean_balance"] > credit_card_stats["mean_balance"], \
            "Mortgage mean balances should be higher than credit card"

        assert mortgage_stats["median_balance"] > credit_card_stats["median_balance"], \
            "Mortgage median balances should be higher than credit card"

        assert mortgage_stats["max_balance"] > credit_card_stats["max_balance"], \
            "Mortgage max balances should be higher than credit card"

def test_portfolio_error_handling():
    """Test error handling in portfolio generation."""
    # Test with invalid n_loans
    with pytest.raises(ValueError, match="must be positive"):
        generate_base_portfolio(n_loans=-100)

    # Test with zero n_loans
    with pytest.raises(ValueError, match="must be positive"):
        generate_base_portfolio(n_loans=0)

def test_portfolio_metadata_completeness():
    """Test that portfolio metadata is complete and accurate."""
    result = generate_portfolio_with_statistics(n_loans=1000, seed=123, include_all_products=False)

    metadata = result["metadata"]

    # Check all expected metadata fields
    expected_fields = {"n_loans", "seed", "include_all_products", "generation_timestamp", "product_distribution"}
    assert set(metadata.keys()) == expected_fields, \
        f"Metadata should contain all expected fields: {expected_fields}"

    # Check field values
    assert metadata["n_loans"] == 1000, "n_loans should match"
    assert metadata["seed"] == 123, "seed should match"
    assert metadata["include_all_products"] is False, "include_all_products should match"

    # Check that product distribution sums to 1
    product_dist = metadata["product_distribution"]
    assert abs(sum(product_dist.values()) - 1.0) < 1e-6, "Product distribution should sum to 1"

    # Check that generation timestamp is recent
    timestamp = metadata["generation_timestamp"]
    assert isinstance(timestamp, pd.Timestamp), "Timestamp should be pandas Timestamp"
    assert timestamp > pd.Timestamp("2020-01-01"), "Timestamp should be reasonable"