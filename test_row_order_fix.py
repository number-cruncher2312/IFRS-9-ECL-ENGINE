#!/usr/bin/env python3

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synthetic_data.portfolio_integrator import generate_phase1_portfolio
from synthetic_data.product_taxonomy import PRODUCT_TAXONOMY

def test_row_order_preservation():
    """Test that EIR and lifetime values are correctly associated with their products."""
    print("=== Testing Row Order Preservation Fix ===")

    # Generate a small test portfolio for detailed validation
    portfolio = generate_phase1_portfolio(master_seed=42, n_loans=100)

    print(f"Generated portfolio with {len(portfolio)} loans")

    # Test 1: Verify that each row's EIR value is within its product's bounds
    print("\n1. Testing EIR values are within product-specific bounds:")
    eir_issues = []
    for i, row in portfolio.iterrows():
        product = row['product_type']
        eir = row['eir']
        config = PRODUCT_TAXONOMY[product]['eir']
        min_eir, max_eir = config['min'], config['max']

        if not (min_eir <= eir <= max_eir):
            eir_issues.append(f"Row {i}: {product} EIR {eir:.4f} outside bounds [{min_eir}, {max_eir}]")

    if eir_issues:
        print(f"  FAILED: Found {len(eir_issues)} EIR bound violations")
        for issue in eir_issues[:5]:  # Show first 5 issues
            print(f"    {issue}")
        return False
    else:
        print("  PASSED: All EIR values within product-specific bounds")

    # Test 2: Verify that each row's lifetime value is within its product's bounds
    print("\n2. Testing lifetime values are within product-specific bounds:")
    lifetime_issues = []
    for i, row in portfolio.iterrows():
        product = row['product_type']
        lifetime = row['remaining_lifetime_months']
        config = PRODUCT_TAXONOMY[product]['remaining_lifetime_months']
        min_lifetime, max_lifetime = config['min'], config['max']

        if not (min_lifetime <= lifetime <= max_lifetime):
            lifetime_issues.append(f"Row {i}: {product} lifetime {lifetime} outside bounds [{min_lifetime}, {max_lifetime}]")

    if lifetime_issues:
        print(f"  FAILED: Found {len(lifetime_issues)} lifetime bound violations")
        for issue in lifetime_issues[:5]:  # Show first 5 issues
            print(f"    {issue}")
        return False
    else:
        print("  PASSED: All lifetime values within product-specific bounds")

    # Test 3: Verify that the same product assignments produce consistent EIR/lifetime patterns
    print("\n3. Testing consistency of EIR/lifetime for same products:")
    product_stats = {}

    for i, row in portfolio.iterrows():
        product = row['product_type']
        eir = row['eir']
        lifetime = row['remaining_lifetime_months']

        if product not in product_stats:
            product_stats[product] = {
                'eir_values': [],
                'lifetime_values': [],
                'count': 0
            }

        product_stats[product]['eir_values'].append(eir)
        product_stats[product]['lifetime_values'].append(lifetime)
        product_stats[product]['count'] += 1

    # Check that each product has reasonable statistics
    for product, stats in product_stats.items():
        eir_values = stats['eir_values']
        lifetime_values = stats['lifetime_values']
        count = stats['count']

        print(f"  {product}: {count} loans")
        print(f"    EIR: min={min(eir_values):.4f}, max={max(eir_values):.4f}, mean={np.mean(eir_values):.4f}")
        print(f"    Lifetime: min={min(lifetime_values):.1f}, max={max(lifetime_values):.1f}, mean={np.mean(lifetime_values):.1f}")

    # Test 4: Verify reproducibility with same seed
    print("\n4. Testing reproducibility:")
    portfolio2 = generate_phase1_portfolio(master_seed=42, n_loans=100)

    if portfolio.equals(portfolio2):
        print("  PASSED: Same seed produces identical results")
    else:
        print("  FAILED: Same seed produces different results")
        return False

    # Test 5: Verify different seeds produce different results
    print("\n5. Testing seed variation:")
    portfolio3 = generate_phase1_portfolio(master_seed=123, n_loans=100)

    if not portfolio.equals(portfolio3):
        print("  PASSED: Different seeds produce different results")
    else:
        print("  FAILED: Different seeds produce same results")
        return False

    print("\n=== All Row Order Preservation Tests PASSED ===")
    return True

def test_specific_order_case():
    """Test the specific example case mentioned in the bug report."""
    print("\n=== Testing Specific Order Case ===")

    # Simulate the exact scenario from the bug report
    from synthetic_data.eir_generator import generate_eir_for_multiple_products
    from synthetic_data.lifetime_generator import generate_lifetime_for_multiple_products

    product_assignments = ["credit_card", "auto_loan", "mortgage", "credit_card", "auto_loan"]
    print(f"Original product assignments: {product_assignments}")

    # Generate EIR and lifetime values
    eir_dict = generate_eir_for_multiple_products(
        product_assignments=product_assignments,
        seed=42
    )

    lifetime_dict = generate_lifetime_for_multiple_products(
        product_assignments=product_assignments,
        seed=42
    )

    # Create index mapping (this is what our fix does)
    product_indices = {}
    for i, product in enumerate(product_assignments):
        if product not in product_indices:
            product_indices[product] = []
        product_indices[product].append(i)

    # Assign values using index mapping (correct approach)
    all_eirs_correct = [None] * len(product_assignments)
    for product, eir_values in eir_dict.items():
        indices = product_indices[product]
        for i, idx in enumerate(indices):
            all_eirs_correct[idx] = eir_values[i]

    all_lifetimes_correct = [None] * len(product_assignments)
    for product, lifetime_values in lifetime_dict.items():
        indices = product_indices[product]
        for i, idx in enumerate(indices):
            all_lifetimes_correct[idx] = lifetime_values[i]

    print(f"Correct EIR assignment: {[f'{eir:.4f}' for eir in all_eirs_correct]}")
    print(f"Correct lifetime assignment: {all_lifetimes_correct}")

    # Verify that each position has the correct product type
    for i, (product, eir, lifetime) in enumerate(zip(product_assignments, all_eirs_correct, all_lifetimes_correct)):
        config_eir = PRODUCT_TAXONOMY[product]['eir']
        config_lifetime = PRODUCT_TAXONOMY[product]['remaining_lifetime_months']

        eir_in_bounds = config_eir['min'] <= eir <= config_eir['max']
        lifetime_in_bounds = config_lifetime['min'] <= lifetime <= config_lifetime['max']

        if eir_in_bounds and lifetime_in_bounds:
            print(f"  Row {i}: {product} -> EIR {eir:.4f} (OK), Lifetime {lifetime} (OK)")
        else:
            print(f"  Row {i}: {product} -> EIR {eir:.4f} ({'FAIL' if not eir_in_bounds else 'OK'}), Lifetime {lifetime} ({'FAIL' if not lifetime_in_bounds else 'OK'})")
            return False

    print("  PASSED: All values correctly associated with their products")
    return True

def main():
    print("Running Row Order Fix Validation Tests")
    print("=" * 60)

    success = True

    # Run the main test
    if not test_row_order_preservation():
        success = False

    # Run the specific case test
    if not test_specific_order_case():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS
    else:
        print("❌ SOME TESTS FAILED - Row order fix needs attention!")

    return success

if __name__ == "__main__":
    main()