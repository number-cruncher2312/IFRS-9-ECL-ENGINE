"""
V1 Synthetic LGD Generator
===========================

Deterministic LGD lookup for IFRS 9 synthetic loan portfolio.

This module implements the V1 LGD methodology as a simple product-based lookup.
LGD is the loss severity conditional on default (0 <= LGD <= 1).

Key design principles:
- LGD is NOT an ML model or stochastic generator
- LGD is a documented product-based lookup assumption
- Deterministic: same product always returns same LGD
- Uses existing product taxonomy to avoid duplication
- Validates that returned LGDs are within [0, 1]
- Rejects unknown product types clearly

Methodological notes:
- LGD values are SYNTHETIC V1 ASSUMPTIONS, not empirical/calibrated estimates
- LGD is conditional on default: EAD * LGD = expected loss severity given default
- Does NOT use PD, EAD, default status, EIR, lifetime, or borrower characteristics
- Does NOT implement collateral-value modelling, recovery curves, discounting, etc.
- Preserves distinction between product, LGD category, and LGD rate

V1 active products (assigned in V1 portfolio):
- credit_card
- auto_loan
- mortgage

V1 inactive products (defined but not assigned in V1):
- student_loan
- other_personal_loan
"""

from typing import Dict, Any, Union, List, Optional
import pandas as pd
import numpy as np

from synthetic_data.product_taxonomy import (
    PRODUCT_TYPES,
    PRODUCT_TAXONOMY,
    LGD_RATES,
    LGD_CATEGORIES,
    get_lgd_rate as get_lgd_rate_from_taxonomy
)

# ---------------------------------------------------------------------------
# Core LGD lookup functions
# ---------------------------------------------------------------------------

def get_lgd_rate(product_type: str) -> float:
    """
    Return the synthetic V1 LGD rate for a single product type.

    Args:
        product_type: String product type (e.g., "mortgage", "credit_card")

    Returns:
        float: LGD rate in [0, 1]

    Raises:
        ValueError: If product_type is unknown or not in PRODUCT_TYPES
        AssertionError: If the returned LGD is outside [0, 1] (internal error)
    """
    # Validate input
    if product_type not in PRODUCT_TYPES:
        raise ValueError(
            f"Unknown product_type: {product_type!r}. "
            f"Valid types: {sorted(PRODUCT_TYPES)}"
        )

    # Get LGD rate from taxonomy
    lgd_rate = get_lgd_rate_from_taxonomy(product_type)

    # Validate LGD is in valid range
    assert 0.0 <= lgd_rate <= 1.0, (
        f"LGD rate {lgd_rate} for product {product_type!r} is outside [0, 1]. "
        f"This indicates an internal error in the taxonomy."
    )

    return lgd_rate

def get_lgd_rates_for_products(product_types: List[str]) -> Dict[str, float]:
    """
    Return LGD rates for multiple product types.

    Args:
        product_types: List of product type strings

    Returns:
        Dict[str, float]: Mapping of product_type -> LGD rate

    Raises:
        ValueError: If any product_type is unknown
    """
    result = {}
    for product_type in product_types:
        result[product_type] = get_lgd_rate(product_type)
    return result

def assign_lgd_to_dataframe(df: pd.DataFrame, product_column: str = "product_type") -> pd.DataFrame:
    """
    Assign LGD rates to a DataFrame containing product types.

    Args:
        df: DataFrame containing product types
        product_column: Name of column containing product types (default: "product_type")

    Returns:
        DataFrame with added "lgd" column

    Raises:
        ValueError: If product_column contains unknown product types
        KeyError: If product_column does not exist in DataFrame
    """
    if product_column not in df.columns:
        raise KeyError(f"Column {product_column!r} not found in DataFrame")

    # Validate all products are known
    unknown_products = set(df[product_column]) - PRODUCT_TYPES
    if unknown_products:
        raise ValueError(
            f"Unknown product types in DataFrame: {sorted(unknown_products)}. "
            f"Valid types: {sorted(PRODUCT_TYPES)}"
        )

    # Assign LGD rates
    df = df.copy()
    df["lgd"] = df[product_column].apply(get_lgd_rate)

    return df

# ---------------------------------------------------------------------------
# Product and LGD category information
# ---------------------------------------------------------------------------

def get_product_lgd_category(product_type: str) -> str:
    """
    Return the LGD category for a product type.

    Args:
        product_type: String product type

    Returns:
        str: LGD category (e.g., "LGD_MORTGAGE")

    Raises:
        ValueError: If product_type is unknown
    """
    if product_type not in PRODUCT_TYPES:
        raise ValueError(
            f"Unknown product_type: {product_type!r}. "
            f"Valid types: {sorted(PRODUCT_TYPES)}"
        )
    return PRODUCT_TAXONOMY[product_type]["lgd_category"]

def get_lgd_category_rate(lgd_category: str) -> float:
    """
    Return the LGD rate for a specific LGD category.

    Args:
        lgd_category: String LGD category (e.g., "LGD_MORTGAGE")

    Returns:
        float: LGD rate in [0, 1]

    Raises:
        ValueError: If lgd_category is unknown
    """
    if lgd_category not in LGD_CATEGORIES:
        raise ValueError(
            f"Unknown lgd_category: {lgd_category!r}. "
            f"Valid categories: {sorted(LGD_CATEGORIES)}"
        )

    rate = LGD_RATES[lgd_category]

    # Validate rate is in valid range
    assert 0.0 <= rate <= 1.0, (
        f"LGD rate {rate} for category {lgd_category!r} is outside [0, 1]. "
        f"This indicates an internal error in the taxonomy."
    )

    return rate

# ---------------------------------------------------------------------------
# V1 product mix information
# ---------------------------------------------------------------------------

def get_v1_active_products() -> List[str]:
    """
    Return list of V1 active products (assigned in V1 portfolio).

    Returns:
        List[str]: Active product types
    """
    return ["credit_card", "auto_loan", "mortgage"]

def get_v1_inactive_products() -> List[str]:
    """
    Return list of V1 inactive products (defined but not assigned in V1).

    Returns:
        List[str]: Inactive product types
    """
    return ["student_loan", "other_personal_loan"]

def get_all_product_lgd_mappings() -> Dict[str, Dict[str, float]]:
    """
    Return complete mapping of all products to their LGD categories and rates.

    Returns:
        Dict[str, Dict[str, float]]: Mapping of product_type -> {
            "lgd_category": str,
            "lgd_rate": float
        }
    """
    result = {}
    for product_type in PRODUCT_TYPES:
        lgd_category = PRODUCT_TAXONOMY[product_type]["lgd_category"]
        lgd_rate = LGD_RATES[lgd_category]
        result[product_type] = {
            "lgd_category": lgd_category,
            "lgd_rate": lgd_rate
        }
    return result

# ---------------------------------------------------------------------------
# Internal validation tests
# ---------------------------------------------------------------------------

def _run_internal_validation() -> None:
    """Validate LGD generator invariants. Raises AssertionError on failure."""
    # 1. All products return valid LGD rates
    for product_type in PRODUCT_TYPES:
        lgd_rate = get_lgd_rate(product_type)
        assert 0.0 <= lgd_rate <= 1.0, (
            f"{product_type}: LGD rate {lgd_rate} outside [0, 1]"
        )

    # 2. LGD categories match taxonomy
    for product_type in PRODUCT_TYPES:
        category = get_product_lgd_category(product_type)
        assert category in LGD_CATEGORIES, (
            f"{product_type}: LGD category {category} not in LGD_CATEGORIES"
        )

    # 3. Category rates match direct lookup
    for product_type in PRODUCT_TYPES:
        category = get_product_lgd_category(product_type)
        category_rate = get_lgd_category_rate(category)
        product_rate = get_lgd_rate(product_type)
        assert category_rate == product_rate, (
            f"{product_type}: category rate {category_rate} != product rate {product_rate}"
        )

    # 4. V1 active/inactive products are correct
    active = get_v1_active_products()
    inactive = get_v1_inactive_products()
    assert set(active) == {"credit_card", "auto_loan", "mortgage"}, (
        f"Active products incorrect: {active}"
    )
    assert set(inactive) == {"student_loan", "other_personal_loan"}, (
        f"Inactive products incorrect: {inactive}"
    )
    assert set(active) | set(inactive) == PRODUCT_TYPES, (
        f"Active + inactive != all products"
    )
    assert len(set(active) & set(inactive)) == 0, (
        f"Active and inactive products overlap"
    )

    # 5. Error handling works
    with pytest.raises(ValueError):
        get_lgd_rate("unknown_product")

    with pytest.raises(ValueError):
        get_lgd_category_rate("unknown_category")

if __name__ == "__main__":
    _run_internal_validation()
    print("LGD generator V1 validation: OK")

    # Print mappings
    mappings = get_all_product_lgd_mappings()
    print("\nProduct -> LGD Mappings:")
    for product, info in mappings.items():
        print(f"  {product:20} -> {info['lgd_category']:15} = {info['lgd_rate']:.2f}")

    print(f"\nV1 Active Products: {get_v1_active_products()}")
    print(f"V1 Inactive Products: {get_v1_inactive_products()}")