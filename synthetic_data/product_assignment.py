"""
Product Assignment Module for Synthetic Loan Dataset
===================================================

This module implements the product-assignment methodology for the synthetic loan dataset,
calibrated to authoritative U.S. account-count data.

Current Product Mix (V1)
------------------------
The initial product-mix calculation uses ONLY three categories with directly comparable
account-count data:

* Credit card: ~74.82% (643.3 million accounts)
* Auto: ~12.80% (110.0 million accounts)
* Mortgage + HELOC: ~12.38% (106.4 million accounts)

These sum to 859.7 million directly comparable accounts.

Important Notes
---------------
1. Student loans are NOT mechanically included in this calculation. The available
   student-loan figure is a borrower/recipient count rather than a directly
   comparable account count, so we do not assume that each borrower has 3-4 accounts
   or otherwise invent a conversion factor.

2. "Other personal loans" are likewise not included due to lack of authoritative
   account-level estimates.

3. The architecture is designed to be flexible enough that student loans and other
   personal loans can be added later if we obtain defensible account-level estimates.

Methodology
-----------
This is an observed-account-based approximation, NOT a claim that these percentages
represent the entire U.S. consumer-loan market. The probabilities are derived from
authoritative account-count data and are used to generate synthetic loan portfolios
that reflect the relative prevalence of different product types.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from .product_taxonomy import PRODUCT_TYPES

# ---------------------------------------------------------------------------
# V1 Active Product Probabilities
# Based on authoritative U.S. account-count data
# ---------------------------------------------------------------------------

# V1 ACTIVE products with defensible account-count data
V1_ACTIVE_PRODUCT_PROBABILITIES: Dict[str, float] = {
    "credit_card": 0.7482,      # 643.3M / 859.7M accounts
    "auto_loan": 0.1280,        # 110.0M / 859.7M accounts
    "mortgage": 0.1238,         # 106.4M / 859.7M accounts (includes HELOC)
}

# V1 INACTIVE products (taxonomy defined but no assignment probability)
# These products are defined in the taxonomy but have NO V1 assignment probability
# due to lack of authoritative account-level estimates.
V1_INACTIVE_PRODUCTS: set = {
    "student_loan",             # Borrower/recipient count only, not account count
    "other_personal_loan"       # No authoritative account-level estimates available
}

# Default product probabilities for V1 (only active products)
DEFAULT_PRODUCT_PROBABILITIES: Dict[str, float] = V1_ACTIVE_PRODUCT_PROBABILITIES.copy()
-------

# ---------------------------------------------------------------------------
# Product Assignment Functions
# Note: V1 assignments use ONLY V1_ACTIVE_PRODUCT_PROBABILITIES
#       V1_INACTIVE_PRODUCTS are NOT assigned in V1
# ---------------------------------------------------------------------------

def validate_product_probabilities(probabilities: Dict[str, float]) -> None:
    """
    Validate that product probabilities are properly configured.

    Args:
        probabilities: Dictionary mapping product types to probabilities

    Raises:
        ValueError: If probabilities are invalid (negative, sum != 1, unknown products)
    """
    # Check that all probabilities are non-negative
    for product, prob in probabilities.items():
        if prob < 0:
            raise ValueError(f"Probability for {product} is negative: {prob}")
        if prob > 1:
            raise ValueError(f"Probability for {product} exceeds 1: {prob}")

    # Check that all products are known
    unknown_products = set(probabilities.keys()) - PRODUCT_TYPES
    if unknown_products:
        raise ValueError(f"Unknown product types: {unknown_products}. Valid types: {PRODUCT_TYPES}")

    # Check that probabilities sum to 1 (within floating-point tolerance)
    total = sum(probabilities.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Product probabilities sum to {total}, expected 1.0")

def assign_products(
    n_borrowers: int,
    probabilities: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    include_all_products: bool = False
) -> List[str]:
    """
    Assign product types to borrowers using configured probabilities.

    Args:
        n_borrowers: Number of borrowers to assign products to
        probabilities: Optional custom product probabilities. If None, uses DEFAULT_PRODUCT_PROBABILITIES.
        seed: Random seed for reproducibility
        include_all_products: If True, includes all products from PRODUCT_TYPES with equal
                            probabilities for any products not in the probabilities dict.
                            If False (default), only uses products specified in probabilities.

    Returns:
        List of product assignments (one per borrower)

    Raises:
        ValueError: If probabilities are invalid or configuration is inconsistent
    """
    # Use default probabilities if none provided
    if probabilities is None:
        probabilities = DEFAULT_PRODUCT_PROBABILITIES.copy()
    else:
        probabilities = probabilities.copy()

    # Validate the probabilities
    validate_product_probabilities(probabilities)

    # Handle the include_all_products option
    if include_all_products:
        # Add any missing products with equal probabilities
        missing_products = PRODUCT_TYPES - set(probabilities.keys())
        if missing_products:
            # Calculate remaining probability to distribute equally
            current_total = sum(probabilities.values())
            remaining_prob = 1.0 - current_total
            equal_share = remaining_prob / len(missing_products)

            for product in missing_products:
                probabilities[product] = equal_share

            # Re-validate after adding missing products
            validate_product_probabilities(probabilities)

    # Extract product names and probabilities for sampling
    products = list(probabilities.keys())
    probs = list(probabilities.values())

    # Normalize probabilities to handle any floating-point issues
    probs = np.array(probs) / np.sum(probs)

    # Generate random assignments
    rng = np.random.default_rng(seed)
    assignments = rng.choice(products, size=n_borrowers, p=probs)

    return list(assignments)

def assign_products_to_dataframe(
    df: pd.DataFrame,
    probabilities: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    include_all_products: bool = False,
    product_column: str = "product_type"
) -> pd.DataFrame:
    """
    Assign product types to a dataframe of borrowers.

    Args:
        df: Input dataframe (should contain borrower data)
        probabilities: Optional custom product probabilities
        seed: Random seed for reproducibility
        include_all_products: Whether to include all product types
        product_column: Name of column to add for product assignments

    Returns:
        DataFrame with added product_type column
    """
    # Assign products
    n_borrowers = len(df)
    product_assignments = assign_products(
        n_borrowers=n_borrowers,
        probabilities=probabilities,
        seed=seed,
        include_all_products=include_all_products
    )

    # Add to dataframe
    result_df = df.copy()
    result_df[product_column] = product_assignments

    return result_df

# ---------------------------------------------------------------------------
# Validation and Testing Functions
# ---------------------------------------------------------------------------

def validate_product_distribution(
    product_assignments: List[str],
    expected_probabilities: Dict[str, float],
    tolerance: float = 0.05,
    min_sample_size: int = 1000
) -> Dict[str, float]:
    """
    Validate that generated product assignments approximately follow expected distribution.

    Args:
        product_assignments: List of product assignments to validate
        expected_probabilities: Expected probabilities for each product
        tolerance: Allowed deviation from expected probabilities
        min_sample_size: Minimum sample size required for meaningful validation

    Returns:
        Dictionary of actual probabilities observed in the sample

    Raises:
        ValueError: If sample size is too small or distribution deviates too much
    """
    if len(product_assignments) < min_sample_size:
        raise ValueError(f"Sample size {len(product_assignments)} is too small for validation. "
                        f"Minimum required: {min_sample_size}")

    # Calculate actual distribution
    actual_counts = pd.Series(product_assignments).value_counts(normalize=True)
    actual_probs = actual_counts.to_dict()

    # Check each product's probability
    for product, expected_prob in expected_probabilities.items():
        actual_prob = actual_probs.get(product, 0.0)
        deviation = abs(actual_prob - expected_prob)

        if deviation > tolerance:
            raise ValueError(
                f"Product {product}: actual probability {actual_prob:.4f} "
                f"deviates from expected {expected_prob:.4f} by {deviation:.4f} "
                f"(tolerance: {tolerance})"
            )

    return actual_probs

# ---------------------------------------------------------------------------
# Configuration Management
# ---------------------------------------------------------------------------

def get_v1_active_products() -> Dict[str, float]:
    """
    Get the V1 active product probabilities.

    Returns:
        Copy of the V1 active product probabilities dictionary
    """
    return V1_ACTIVE_PRODUCT_PROBABILITIES.copy()

def get_v1_inactive_products() -> set:
    """
    Get the V1 inactive products (defined but not assigned).

    Returns:
        Set of V1 inactive product names
    """
    return V1_INACTIVE_PRODUCTS.copy()

def get_default_product_probabilities() -> Dict[str, float]:
    """
    Get the default product probabilities.

    Returns:
        Copy of the default product probabilities dictionary
    """
    return DEFAULT_PRODUCT_PROBABILITIES.copy()

def set_default_product_probabilities(new_probabilities: Dict[str, float]) -> None:
    """
    Update the default product probabilities (use with caution).

    Args:
        new_probabilities: New default probabilities to use

    Raises:
        ValueError: If the new probabilities are invalid
    """
    global DEFAULT_PRODUCT_PROBABILITIES
    validate_product_probabilities(new_probabilities)
    DEFAULT_PRODUCT_PROBABILITIES = new_probabilities.copy()

# ---------------------------------------------------------------------------
# Internal Validation Tests
# ---------------------------------------------------------------------------

def _run_validation_tests() -> None:
    """Run internal validation tests for the product assignment module."""
    # Test 1: Default probabilities sum to 1
    assert np.isclose(sum(DEFAULT_PRODUCT_PROBABILITIES.values()), 1.0), \
        "Default probabilities do not sum to 1"

    # Test 2: Default probabilities are non-negative
    for prob in DEFAULT_PRODUCT_PROBABILITIES.values():
        assert prob >= 0, f"Negative probability found: {prob}"

    # Test 3: All default products are valid
    for product in DEFAULT_PRODUCT_PROBABILITIES.keys():
        assert product in PRODUCT_TYPES, f"Unknown product in defaults: {product}"

    # Test 4: Validation function works correctly
    try:
        validate_product_probabilities({"invalid_product": 0.5})
        assert False, "Validation should have failed for unknown product"
    except ValueError:
        pass  # Expected

    try:
        validate_product_probabilities({"credit_card": 1.1})
        assert False, "Validation should have failed for probability > 1"
    except ValueError:
        pass  # Expected

    try:
        validate_product_probabilities({"credit_card": 0.5, "auto_loan": 0.6})
        assert False, "Validation should have failed for sum != 1"
    except ValueError:
        pass  # Expected

    # Test 5: Assignment function works
    assignments = assign_products(100, DEFAULT_PRODUCT_PROBABILITIES, seed=42)
    assert len(assignments) == 100, "Wrong number of assignments"
    assert all(assignment in PRODUCT_TYPES for assignment in assignments), \
        "Invalid product assignments generated"

    print("Product assignment validation: OK")

if __name__ == "__main__":
    _run_validation_tests()
    print("Product assignment module initialized successfully.")
    print(f"Default product probabilities: {DEFAULT_PRODUCT_PROBABILITIES}")