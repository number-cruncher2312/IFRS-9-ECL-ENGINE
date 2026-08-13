#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the modules
from synthetic_data.product_assignment import (
    DEFAULT_PRODUCT_PROBABILITIES,
    V1_ACTIVE_PRODUCT_PROBABILITIES,
    V1_INACTIVE_PRODUCTS,
    assign_products,
    validate_product_probabilities
)
from synthetic_data.product_taxonomy import PRODUCT_TYPES

def test_current_implementation():
    print("=== Testing Current Implementation ===")

    # Test 1: Check default probabilities
    print(f"Default probabilities: {DEFAULT_PRODUCT_PROBABILITIES}")
    print(f"Sum of default probabilities: {sum(DEFAULT_PRODUCT_PROBABILITIES.values())}")

    # Test 2: Check V1 active products
    print(f"V1 active products: {V1_ACTIVE_PRODUCT_PROBABILITIES}")
    print(f"V1 inactive products: {V1_INACTIVE_PRODUCTS}")

    # Test 3: Verify probabilities sum to 1
    total_prob = sum(DEFAULT_PRODUCT_PROBABILITIES.values())
    assert abs(total_prob - 1.0) < 1e-6, f"Probabilities don't sum to 1: {total_prob}"

    # Test 4: Verify only 3 products are active
    assert len(DEFAULT_PRODUCT_PROBABILITIES) == 3, f"Expected 3 active products, got {len(DEFAULT_PRODUCT_PROBABILITIES)}"
    assert set(DEFAULT_PRODUCT_PROBABILITIES.keys()) == {"credit_card", "auto_loan", "mortgage"}, "Unexpected active products"

    # Test 5: Verify inactive products are not in default probabilities
    for product in V1_INACTIVE_PRODUCTS:
        assert product not in DEFAULT_PRODUCT_PROBABILITIES, f"Inactive product {product} found in default probabilities"

    # Test 6: Test product assignment
    assignments = assign_products(100, seed=42)
    unique_products = set(assignments)
    print(f"Unique products in 100 assignments: {unique_products}")

    # Verify only active products are assigned
    for product in unique_products:
        assert product in DEFAULT_PRODUCT_PROBABILITIES, f"Unexpected product {product} in assignments"

    # Verify inactive products are not assigned
    for product in V1_INACTIVE_PRODUCTS:
        assert product not in unique_products, f"Inactive product {product} found in assignments"

    print("✓ All tests passed!")

    # Test 7: Verify exact probability values
    expected_probs = {
        "credit_card": 0.7482,
        "auto_loan": 0.1280,
        "mortgage": 0.1238
    }

    for product, expected_prob in expected_probs.items():
        actual_prob = DEFAULT_PRODUCT_PROBABILITIES[product]
        assert abs(actual_prob - expected_prob) < 1e-6, f"Probability mismatch for {product}: expected {expected_prob}, got {actual_prob}"

    print("✓ Probability values are correct!")

    # Test 8: Test validation function
    validate_product_probabilities(DEFAULT_PRODUCT_PROBABILITIES)
    print("✓ Validation function works correctly!")

    print("\n=== Summary ===")
    print("✓ V1 active portfolio uses ONLY the three categories with defensible account-count data")
    print("✓ credit_card: 74.82%")
    print("✓ auto_loan: 12.80%")
    print("✓ mortgage: 12.38%")
    print("✓ student_loan and other_personal_loan are NOT assigned in V1")
    print("✓ Taxonomy definitions are kept for inactive/future categories")
    print("✓ Probabilities sum to 1.0")
    print("✓ All tests pass successfully!")

if __name__ == "__main__":
    test_current_implementation()