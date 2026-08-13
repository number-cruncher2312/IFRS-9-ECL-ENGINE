"""
Portfolio Generator Module for Synthetic Loan Dataset
====================================================

This module serves as the orchestration layer for generating the 10,000-loan base portfolio
with product assignments, balances, and EAD values.

Key Features:
- Generates 10,000-loan base portfolio by default
- Orchestrates product assignment, balance generation, and EAD calculation
- Uses explicit seeds for full reproducibility
- Preserves existing product taxonomy and assignment methodology
- For V1: EAD = balance (no unused-limit or CCF model)

Portfolio Structure:
- Uses V1 active products only (credit_card, auto_loan, mortgage)
- Generates right-skewed balances using bounded/truncated lognormal approach
- Attaches product, balance, and EAD to each loan
- Returns structured pandas DataFrame

Important Notes:
- Does NOT implement EIR, lifetime, default, staging, LGD, or ECL (V1 scope)
- Does NOT use PD as input (as specified)
- Does NOT implement borrower-specific adjustments (baseline distributions only)
- Taxonomy values remain synthetic V1 assumptions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from .product_assignment_fixed import assign_products, DEFAULT_PRODUCT_PROBABILITIES
from .balance_generator import (
    generate_balances_for_multiple_products,
    generate_ead_from_balances,
    validate_balance_generation
)
from .product_taxonomy import PRODUCT_TYPES

def generate_base_portfolio(
    n_loans: int = 10000,
    seed: Optional[int] = 42,
    include_all_products: bool = False
) -> pd.DataFrame:
    """
    Generate the base portfolio with product assignments, balances, and EAD.

    Args:
        n_loans: Number of loans to generate (default: 10,000)
        seed: Random seed for reproducibility (default: 42)
        include_all_products: If True, includes all products from PRODUCT_TYPES.
                             If False (default), uses only V1 active products.

    Returns:
        DataFrame containing the generated portfolio with columns:
        - loan_id: Unique identifier for each loan
        - product_type: Assigned product type
        - balance: Generated balance amount (USD)
        - ead: Exposure at default (V1: equal to balance)

    Raises:
        ValueError: If n_loans is not positive or other validation fails
    """
    if n_loans <= 0:
        raise ValueError(f"n_loans must be positive, got {n_loans}")

    # Step 1: Assign products to loans
    product_assignments = assign_products(
        n_borrowers=n_loans,
        probabilities=None,  # Use default V1 active product probabilities
        seed=seed,
        include_all_products=include_all_products
    )

    # Step 2: Generate balances for all products
    balances_dict = generate_balances_for_multiple_products(
        product_assignments=product_assignments,
        seed=seed  # Use same seed for reproducibility
    )

    # Step 3: Validate balance generation and get statistics
    balance_stats = validate_balance_generation(balances_dict, product_assignments)

    # Step 4: Generate EAD values (V1: EAD = balance)
    # Flatten all balances and generate corresponding EAD values
    all_balances = []
    all_eads = []
    all_products = []

    for product, balances in balances_dict.items():
        eads = generate_ead_from_balances(balances)
        all_balances.extend(balances)
        all_eads.extend(eads)
        all_products.extend([product] * len(balances))

    # Step 5: Create DataFrame with loan data
    portfolio_df = pd.DataFrame({
        "loan_id": range(1, n_loans + 1),
        "product_type": all_products,
        "balance": all_balances,
        "ead": all_eads
    })

    # Step 6: Validate EAD = balance constraint
    if not np.allclose(portfolio_df["balance"], portfolio_df["ead"]):
        raise ValueError("EAD values do not equal balance values as required for V1")

    return portfolio_df

def generate_portfolio_with_statistics(
    n_loans: int = 10000,
    seed: Optional[int] = 42,
    include_all_products: bool = False
) -> Dict[str, Any]:
    """
    Generate portfolio and return both DataFrame and detailed statistics.

    Args:
        n_loans: Number of loans to generate (default: 10,000)
        seed: Random seed for reproducibility (default: 42)
        include_all_products: Whether to include all product types

    Returns:
        Dictionary containing:
        - portfolio: Generated portfolio DataFrame
        - statistics: Detailed balance statistics by product
        - metadata: Portfolio generation metadata
    """
    # Generate the base portfolio
    portfolio = generate_base_portfolio(
        n_loans=n_loans,
        seed=seed,
        include_all_products=include_all_products
    )

    # Calculate detailed statistics by product
    statistics = {}
    for product in portfolio["product_type"].unique():
        product_data = portfolio[portfolio["product_type"] == product]
        balances = product_data["balance"].values

        statistics[product] = {
            "count": len(balances),
            "min": float(np.min(balances)),
            "median": float(np.median(balances)),
            "mean": float(np.mean(balances)),
            "max": float(np.max(balances)),
            "std": float(np.std(balances)),
            "skewness": float(pd.Series(balances).skew()),
            "sum": float(np.sum(balances))
        }

    # Add metadata
    metadata = {
        "n_loans": n_loans,
        "seed": seed,
        "include_all_products": include_all_products,
        "generation_timestamp": pd.Timestamp.now(),
        "product_distribution": portfolio["product_type"].value_counts(normalize=True).to_dict()
    }

    return {
        "portfolio": portfolio,
        "statistics": statistics,
        "metadata": metadata
    }

def validate_portfolio_reproducibility(
    n_loans: int = 1000,
    seed: int = 42,
    n_trials: int = 3
) -> bool:
    """
    Validate that portfolio generation is reproducible with the same seed.

    Args:
        n_loans: Number of loans for test
        seed: Seed to use for reproducibility test
        n_trials: Number of trials to run

    Returns:
        True if all trials produce identical results, False otherwise
    """
    # Generate multiple portfolios with the same seed
    portfolios = []
    for i in range(n_trials):
        portfolio = generate_base_portfolio(n_loans=n_loans, seed=seed)
        portfolios.append(portfolio)

    # Check that all portfolios are identical
    reference = portfolios[0]
    for i, portfolio in enumerate(portfolios[1:], 1):
        if not portfolio.equals(reference):
            print(f"Reproducibility check failed: trial {i} differs from reference")
            return False

    print(f"Reproducibility validated: {n_trials} trials with seed {seed} produced identical results")
    return True

def validate_portfolio_properties(
    portfolio: pd.DataFrame,
    expected_products: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    Validate key properties of the generated portfolio.

    Args:
        portfolio: Portfolio DataFrame to validate
        expected_products: Optional list of expected product types

    Returns:
        Dictionary of validation results with boolean flags
    """
    validation_results = {}

    # Check required columns
    required_columns = {"loan_id", "product_type", "balance", "ead"}
    validation_results["has_required_columns"] = required_columns.issubset(set(portfolio.columns))

    # Check loan_id uniqueness and sequencing
    validation_results["loan_ids_unique"] = portfolio["loan_id"].is_unique
    validation_results["loan_ids_sequential"] = (portfolio["loan_id"] == range(1, len(portfolio) + 1)).all()

    # Check positive balances
    validation_results["all_balances_positive"] = (portfolio["balance"] > 0).all()

    # Check EAD = balance
    validation_results["ead_equals_balance"] = np.allclose(portfolio["balance"], portfolio["ead"])

    # Check product types
    if expected_products:
        validation_results["expected_products_only"] = set(portfolio["product_type"].unique()).issubset(set(expected_products))
    else:
        validation_results["expected_products_only"] = True  # Skip if not specified

    # Check right-skewness (skewness > 0 for each product)
    skewness_by_product = {}
    for product in portfolio["product_type"].unique():
        product_balances = portfolio[portfolio["product_type"] == product]["balance"]
        skewness = pd.Series(product_balances).skew()
        skewness_by_product[product] = skewness > 0

    validation_results["right_skewness_by_product"] = skewness_by_product

    return validation_results

# Internal validation tests
def _run_validation_tests() -> None:
    """Run internal validation tests for the portfolio generator."""
    print("Running portfolio generator validation tests...")

    # Test 1: Basic portfolio generation
    small_portfolio = generate_base_portfolio(n_loans=100, seed=42)
    assert len(small_portfolio) == 100, "Should generate correct number of loans"
    assert "loan_id" in small_portfolio.columns, "Should have loan_id column"
    assert "product_type" in small_portfolio.columns, "Should have product_type column"
    assert "balance" in small_portfolio.columns, "Should have balance column"
    assert "ead" in small_portfolio.columns, "Should have ead column"

    # Test 2: EAD = balance validation
    assert np.allclose(small_portfolio["balance"], small_portfolio["ead"]), "EAD should equal balance"

    # Test 3: Product distribution (should use V1 active products by default)
    product_counts = small_portfolio["product_type"].value_counts()
    expected_products = set(DEFAULT_PRODUCT_PROBABILITIES.keys())
    assert set(product_counts.index).issubset(expected_products), "Should only use V1 active products by default"

    # Test 4: Reproducibility
    portfolio1 = generate_base_portfolio(n_loans=50, seed=123)
    portfolio2 = generate_base_portfolio(n_loans=50, seed=123)
    assert portfolio1.equals(portfolio2), "Same seed should produce identical portfolios"

    # Test 5: Different seeds produce different results
    portfolio3 = generate_base_portfolio(n_loans=50, seed=456)
    assert not portfolio1.equals(portfolio3), "Different seeds should produce different portfolios"

    # Test 6: Statistics generation
    result = generate_portfolio_with_statistics(n_loans=100, seed=42)
    assert "portfolio" in result, "Should return portfolio"
    assert "statistics" in result, "Should return statistics"
    assert "metadata" in result, "Should return metadata"
    assert len(result["statistics"]) > 0, "Should have statistics for at least one product"

    # Test 7: Validation functions
    validation = validate_portfolio_properties(small_portfolio)
    assert validation["has_required_columns"], "Should have required columns"
    assert validation["all_balances_positive"], "All balances should be positive"
    assert validation["ead_equals_balance"], "EAD should equal balance"

    # Test 8: Reproducibility validation function
    assert validate_portfolio_reproducibility(n_loans=50, seed=789, n_trials=2), "Should be reproducible"

    print("Portfolio generator validation: OK")

if __name__ == "__main__":
    _run_validation_tests()
    print("Portfolio generator module initialized successfully.")