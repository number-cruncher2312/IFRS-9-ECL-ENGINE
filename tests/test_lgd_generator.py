"""
Test suite for V1 LGD Generator
================================

Comprehensive tests for the synthetic_data.lgd_generator module.

Tests cover:
- Correct LGD for each product
- Correct mapping through the taxonomy
- All returned values in [0,1]
- Unknown-product error handling
- Dataframe/batch assignment
- Active V1 products only when testing the V1 portfolio
- Inactive products remain accessible but not assigned by V1 product-assignment
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add the parent directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.lgd_generator import (
    get_lgd_rate,
    get_lgd_rates_for_products,
    assign_lgd_to_dataframe,
    get_product_lgd_category,
    get_lgd_category_rate,
    get_v1_active_products,
    get_v1_inactive_products,
    get_all_product_lgd_mappings
)
from synthetic_data.product_taxonomy import (
    PRODUCT_TYPES,
    LGD_RATES,
    LGD_CATEGORIES,
    PRODUCT_TAXONOMY
)

# ---------------------------------------------------------------------------
# Test data setup
# ---------------------------------------------------------------------------

# Expected LGD rates from taxonomy
EXPECTED_LGD_RATES = {
    "mortgage": 0.10,
    "auto_loan": 0.25,
    "credit_card": 0.55,
    "student_loan": 0.20,
    "other_personal_loan": 0.45
}

# Expected LGD categories from taxonomy
EXPECTED_LGD_CATEGORIES = {
    "mortgage": "LGD_MORTGAGE",
    "auto_loan": "LGD_AUTO",
    "credit_card": "LGD_CREDIT_CARD",
    "student_loan": "LGD_STUDENT",
    "other_personal_loan": "LGD_UNSECURED"
}

# ---------------------------------------------------------------------------
# Core functionality tests
# ---------------------------------------------------------------------------

def test_correct_lgd_for_each_product():
    """Test that each product returns the correct LGD rate."""
    for product, expected_rate in EXPECTED_LGD_RATES.items():
        actual_rate = get_lgd_rate(product)
        assert actual_rate == expected_rate, (
            f"{product}: expected LGD {expected_rate}, got {actual_rate}"
        )

def test_correct_lgd_categories():
    """Test that each product maps to the correct LGD category."""
    for product, expected_category in EXPECTED_LGD_CATEGORIES.items():
        actual_category = get_product_lgd_category(product)
        assert actual_category == expected_category, (
            f"{product}: expected category {expected_category}, got {actual_category}"
        )

def test_all_lgd_values_in_valid_range():
    """Test that all returned LGD values are within [0, 1]."""
    for product in PRODUCT_TYPES:
        lgd_rate = get_lgd_rate(product)
        assert 0.0 <= lgd_rate <= 1.0, (
            f"{product}: LGD rate {lgd_rate} outside [0, 1]"
        )

def test_unknown_product_error_handling():
    """Test that unknown product types raise clear ValueError."""
    with pytest.raises(ValueError, match="Unknown product_type"):
        get_lgd_rate("unknown_product")

    with pytest.raises(ValueError, match="Unknown product_type"):
        get_product_lgd_category("unknown_product")

def test_unknown_lgd_category_error_handling():
    """Test that unknown LGD categories raise clear ValueError."""
    with pytest.raises(ValueError, match="Unknown lgd_category"):
        get_lgd_category_rate("unknown_category")

# ---------------------------------------------------------------------------
# Batch/DataFrame assignment tests
# ---------------------------------------------------------------------------

def test_lgd_rates_for_multiple_products():
    """Test batch assignment for multiple product types."""
    products = ["mortgage", "credit_card", "auto_loan"]
    expected = {
        "mortgage": 0.10,
        "credit_card": 0.55,
        "auto_loan": 0.25
    }

    result = get_lgd_rates_for_products(products)
    assert result == expected, (
        f"Batch assignment failed: expected {expected}, got {result}"
    )

def test_dataframe_assignment_basic():
    """Test DataFrame assignment with valid products."""
    df = pd.DataFrame({
        "product_type": ["mortgage", "credit_card", "auto_loan", "mortgage"]
    })

    result = assign_lgd_to_dataframe(df)
    expected_lgds = [0.10, 0.55, 0.25, 0.10]

    assert "lgd" in result.columns, "LGD column not added to DataFrame"
    assert list(result["lgd"]) == expected_lgds, (
        f"DataFrame LGD assignment failed: expected {expected_lgds}, got {list(result['lgd'])}"
    )

def test_dataframe_assignment_with_unknown_product():
    """Test that DataFrame with unknown products raises ValueError."""
    df = pd.DataFrame({
        "product_type": ["mortgage", "unknown_product"]
    })

    with pytest.raises(ValueError, match="Unknown product types in DataFrame"):
        assign_lgd_to_dataframe(df)

def test_dataframe_assignment_missing_column():
    """Test that DataFrame without product column raises KeyError."""
    df = pd.DataFrame({
        "wrong_column": ["mortgage", "credit_card"]
    })

    with pytest.raises(KeyError, match="Column 'product_type' not found"):
        assign_lgd_to_dataframe(df, product_column="product_type")

def test_dataframe_assignment_custom_column_name():
    """Test DataFrame assignment with custom column name."""
    df = pd.DataFrame({
        "product": ["mortgage", "credit_card"]
    })

    result = assign_lgd_to_dataframe(df, product_column="product")
    expected_lgds = [0.10, 0.55]

    assert list(result["lgd"]) == expected_lgds, (
        f"Custom column assignment failed: expected {expected_lgds}, got {list(result['lgd'])}"
    )

# ---------------------------------------------------------------------------
# V1 product mix tests
# ---------------------------------------------------------------------------

def test_v1_active_products():
    """Test that V1 active products are correctly identified."""
    active = get_v1_active_products()
    expected_active = ["credit_card", "auto_loan", "mortgage"]

    assert set(active) == set(expected_active), (
        f"Active products incorrect: expected {expected_active}, got {active}"
    )

def test_v1_inactive_products():
    """Test that V1 inactive products are correctly identified."""
    inactive = get_v1_inactive_products()
    expected_inactive = ["student_loan", "other_personal_loan"]

    assert set(inactive) == set(expected_inactive), (
        f"Inactive products incorrect: expected {expected_inactive}, got {inactive}"
    )

def test_active_inactive_partition():
    """Test that active + inactive = all products with no overlap."""
    active = set(get_v1_active_products())
    inactive = set(get_v1_inactive_products())
    all_products = set(PRODUCT_TYPES)

    assert active | inactive == all_products, (
        f"Active + inactive != all products: {active | inactive} != {all_products}"
    )
    assert len(active & inactive) == 0, (
        f"Active and inactive products overlap: {active & inactive}"
    )

def test_inactive_products_still_accessible():
    """Test that inactive products are still accessible in taxonomy."""
    inactive = get_v1_inactive_products()

    for product in inactive:
        # Should be able to get LGD for inactive products
        lgd_rate = get_lgd_rate(product)
        assert 0.0 <= lgd_rate <= 1.0, (
            f"Inactive product {product} has invalid LGD: {lgd_rate}"
        )

        # Should be able to get category for inactive products
        category = get_product_lgd_category(product)
        assert category in LGD_CATEGORIES, (
            f"Inactive product {product} has invalid category: {category}"
        )

# ---------------------------------------------------------------------------
# Integration with taxonomy tests
# ---------------------------------------------------------------------------

def test_lgd_rates_match_taxonomy():
    """Test that LGD rates match the taxonomy definitions."""
    for product in PRODUCT_TYPES:
        # Get rate from generator
        gen_rate = get_lgd_rate(product)

        # Get rate directly from taxonomy
        lgd_category = PRODUCT_TAXONOMY[product]["lgd_category"]
        tax_rate = LGD_RATES[lgd_category]

        assert gen_rate == tax_rate, (
            f"{product}: generator rate {gen_rate} != taxonomy rate {tax_rate}"
        )

def test_lgd_categories_match_taxonomy():
    """Test that LGD categories match the taxonomy definitions."""
    for product in PRODUCT_TYPES:
        # Get category from generator
        gen_category = get_product_lgd_category(product)

        # Get category directly from taxonomy
        tax_category = PRODUCT_TAXONOMY[product]["lgd_category"]

        assert gen_category == tax_category, (
            f"{product}: generator category {gen_category} != taxonomy category {tax_category}"
        )

def test_all_product_lgd_mappings():
    """Test the complete product-LGD mapping function."""
    mappings = get_all_product_lgd_mappings()

    # Should have all products
    assert set(mappings.keys()) == PRODUCT_TYPES, (
        f"Mapping keys {set(mappings.keys())} != all products {PRODUCT_TYPES}"
    )

    # Each mapping should have correct structure
    for product, mapping in mappings.items():
        assert "lgd_category" in mapping, f"{product} missing lgd_category"
        assert "lgd_rate" in mapping, f"{product} missing lgd_rate"

        # Category should match expected
        expected_category = EXPECTED_LGD_CATEGORIES[product]
        assert mapping["lgd_category"] == expected_category, (
            f"{product}: category {mapping['lgd_category']} != expected {expected_category}"
        )

        # Rate should match expected
        expected_rate = EXPECTED_LGD_RATES[product]
        assert mapping["lgd_rate"] == expected_rate, (
            f"{product}: rate {mapping['lgd_rate']} != expected {expected_rate}"
        )

# ---------------------------------------------------------------------------
# Edge case and robustness tests
# ---------------------------------------------------------------------------

def test_empty_product_list():
    """Test behavior with empty product list."""
    result = get_lgd_rates_for_products([])
    assert result == {}, "Empty product list should return empty dict"

def test_single_product_dataframe():
    """Test DataFrame with single product."""
    df = pd.DataFrame({
        "product_type": ["credit_card"]
    })

    result = assign_lgd_to_dataframe(df)
    assert list(result["lgd"]) == [0.55], (
        f"Single product DataFrame failed: expected [0.55], got {list(result['lgd'])}"
    )

def test_large_dataframe_assignment():
    """Test DataFrame assignment with many rows."""
    n = 1000
    products = ["mortgage", "credit_card", "auto_loan"]
    df = pd.DataFrame({
        "product_type": np.random.choice(products, n)
    })

    result = assign_lgd_to_dataframe(df)

    # All LGDs should be valid
    assert all(0.0 <= lgd <= 1.0 for lgd in result["lgd"]), (
        "Large DataFrame contains invalid LGD values"
    )

    # Check a few specific cases
    mortgage_lgds = result[result["product_type"] == "mortgage"]["lgd"]
    assert all(lgd == 0.10 for lgd in mortgage_lgds), (
        "Mortgage LGDs in large DataFrame are incorrect"
    )

# ---------------------------------------------------------------------------
# Deterministic behavior tests
# ---------------------------------------------------------------------------

def test_deterministic_single_product():
    """Test that same product always returns same LGD (deterministic)."""
    for product in PRODUCT_TYPES:
        rate1 = get_lgd_rate(product)
        rate2 = get_lgd_rate(product)
        assert rate1 == rate2, (
            f"{product}: non-deterministic behavior detected"
        )

def test_deterministic_dataframe_assignment():
    """Test that DataFrame assignment is deterministic."""
    df = pd.DataFrame({
        "product_type": ["mortgage", "credit_card", "auto_loan"]
    })

    result1 = assign_lgd_to_dataframe(df)
    result2 = assign_lgd_to_dataframe(df)

    assert list(result1["lgd"]) == list(result2["lgd"]), (
        "DataFrame assignment is non-deterministic"
    )

# ---------------------------------------------------------------------------
# Test suite metadata
# ---------------------------------------------------------------------------

def test_print_all_mappings():
    """Print all product-LGD mappings for verification (not a real test)."""
    print("\n=== V1 LGD Product Mappings ===")
    mappings = get_all_product_lgd_mappings()
    for product, info in sorted(mappings.items()):
        print(f"  {product:20} -> {info['lgd_category']:15} = {info['lgd_rate']:.2f}")

    print(f"\nActive V1 Products: {sorted(get_v1_active_products())}")
    print(f"Inactive Products: {sorted(get_v1_inactive_products())}")
    print("=================================")

if __name__ == "__main__":
    # Run the test that prints mappings
    test_print_all_mappings()

    # Run all tests
    pytest.main([__file__, "-v"])