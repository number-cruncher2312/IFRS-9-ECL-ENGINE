#!/usr/bin/env python3

"""
Verification script to confirm the V1 product assignment methodology
is correctly implemented according to the specified requirements.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_implementation():
    """Verify that the implementation meets all requirements."""

    print("=== V1 Product Assignment Methodology Verification ===\n")

    # Import the modules
    try:
        from synthetic_data.product_assignment import (
            DEFAULT_PRODUCT_PROBABILITIES,
            V1_ACTIVE_PRODUCT_PROBABILITIES,
            V1_INACTIVE_PRODUCTS
        )
        from synthetic_data.product_taxonomy import PRODUCT_TYPES

        print("✓ Successfully imported modules")

    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
        return False

    # Requirement 1: V1 active portfolio should ONLY include three categories
    print(f"\n1. V1 Active Products:")
    print(f"   {V1_ACTIVE_PRODUCT_PROBABILITIES}")

    expected_active_products = {"credit_card", "auto_loan", "mortgage"}
    actual_active_products = set(V1_ACTIVE_PRODUCT_PROBABILITIES.keys())

    if actual_active_products == expected_active_products:
        print("   ✓ Correct: Only three categories with defensible account-count data")
    else:
        print(f"   ✗ Error: Expected {expected_active_products}, got {actual_active_products}")
        return False

    # Requirement 2: Verify exact probability values
    print(f"\n2. Probability Values:")
    expected_probs = {
        "credit_card": 0.7482,
        "auto_loan": 0.1280,
        "mortgage": 0.1238
    }

    all_correct = True
    for product, expected_prob in expected_probs.items():
        actual_prob = V1_ACTIVE_PRODUCT_PROBABILITIES[product]
        if abs(actual_prob - expected_prob) < 1e-6:
            print(f"   ✓ {product}: {actual_prob} (correct)")
        else:
            print(f"   ✗ {product}: {actual_prob} (expected {expected_prob})")
            all_correct = False

    if not all_correct:
        return False

    # Requirement 3: Probabilities should sum to 1.0
    print(f"\n3. Probability Sum:")
    total_prob = sum(V1_ACTIVE_PRODUCT_PROBABILITIES.values())
    print(f"   Sum: {total_prob}")

    if abs(total_prob - 1.0) < 1e-6:
        print("   ✓ Correct: Probabilities sum to 1.0")
    else:
        print(f"   ✗ Error: Probabilities sum to {total_prob}, expected 1.0")
        return False

    # Requirement 4: Default probabilities should match V1 active probabilities
    print(f"\n4. Default Probabilities:")
    if DEFAULT_PRODUCT_PROBABILITIES == V1_ACTIVE_PRODUCT_PROBABILITIES:
        print("   ✓ Correct: Default probabilities match V1 active probabilities")
    else:
        print("   ✗ Error: Default probabilities don't match V1 active probabilities")
        return False

    # Requirement 5: Inactive products should NOT be assigned in V1
    print(f"\n5. Inactive Products:")
    print(f"   {V1_INACTIVE_PRODUCTS}")

    expected_inactive = {"student_loan", "other_personal_loan"}
    if V1_INACTIVE_PRODUCTS == expected_inactive:
        print("   ✓ Correct: student_loan and other_personal_loan are inactive")
    else:
        print(f"   ✗ Error: Expected {expected_inactive}, got {V1_INACTIVE_PRODUCTS}")
        return False

    # Requirement 6: Inactive products should not be in default probabilities
    print(f"\n6. Inactive Products Exclusion:")
    for product in V1_INACTIVE_PRODUCTS:
        if product not in DEFAULT_PRODUCT_PROBABILITIES:
            print(f"   ✓ {product} correctly excluded from default probabilities")
        else:
            print(f"   ✗ {product} incorrectly included in default probabilities")
            return False

    # Requirement 7: Taxonomy should still include all products
    print(f"\n7. Full Taxonomy:")
    print(f"   All product types: {PRODUCT_TYPES}")

    expected_all_products = {"mortgage", "auto_loan", "credit_card", "student_loan", "other_personal_loan"}
    if PRODUCT_TYPES == expected_all_products:
        print("   ✓ Correct: All product types are defined in taxonomy")
    else:
        print(f"   ✗ Error: Expected {expected_all_products}, got {PRODUCT_TYPES}")
        return False

    print(f"\n=== VERIFICATION COMPLETE ===")
    print("✓ V1 active portfolio uses ONLY the three categories with defensible account-count data")
    print("✓ credit_card: 74.82%")
    print("✓ auto_loan: 12.80%")
    print("✓ mortgage: 12.38%")
    print("✓ student_loan and other_personal_loan are NOT assigned in V1")
    print("✓ Taxonomy definitions are kept for inactive/future categories")
    print("✓ Probabilities sum to 1.0")
    print("✓ All requirements are satisfied!")

    return True

if __name__ == "__main__":
    success = verify_implementation()
    sys.exit(0 if success else 1)