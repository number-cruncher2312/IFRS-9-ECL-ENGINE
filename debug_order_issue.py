#!/usr/bin/env python3

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_order_mismatch():
    """Debug the order mismatch issue in the integrator."""
    print("=== Order Mismatch Debug ===")

    from synthetic_data.eir_generator import generate_eir_for_multiple_products
    from synthetic_data.lifetime_generator import generate_lifetime_for_multiple_products
    from synthetic_data.product_taxonomy import PRODUCT_TAXONOMY

    # Simulate the exact scenario from the integrator
    product_assignments = ["credit_card", "auto_loan", "mortgage", "credit_card", "auto_loan"]

    print("1. Product assignments order:")
    print(f"   {product_assignments}")

    print("\n2. EIR generation:")
    eir_dict = generate_eir_for_multiple_products(
        product_assignments=product_assignments,
        seed=42
    )

    print(f"   EIR dict keys: {list(eir_dict.keys())}")
    print(f"   EIR dict key order: {list(eir_dict.keys())}")

    all_eirs = []
    for product in eir_dict.keys():
        eir_values = eir_dict[product]
        print(f"   Processing {product}: {len(eir_values)} values")
        all_eirs.extend(eir_values)

    print(f"   Final EIR list length: {len(all_eirs)}")
    print(f"   Final EIR values: {[f'{eir:.4f}' for eir in all_eirs]}")

    print("\n3. Lifetime generation:")
    lifetime_dict = generate_lifetime_for_multiple_products(
        product_assignments=product_assignments,
        seed=42
    )

    print(f"   Lifetime dict keys: {list(lifetime_dict.keys())}")
    print(f"   Lifetime dict key order: {list(lifetime_dict.keys())}")

    all_lifetimes = []
    for product in lifetime_dict.keys():
        lifetime_values = lifetime_dict[product]
        print(f"   Processing {product}: {len(lifetime_values)} values")
        all_lifetimes.extend(lifetime_values)

    print(f"   Final lifetime list length: {len(all_lifetimes)}")
    print(f"   Final lifetime values: {all_lifetimes}")

    print("\n4. The problem - order mismatch:")
    print("   Original product_assignments order:")
    for i, product in enumerate(product_assignments):
        print(f"     {i}: {product}")

    print("   EIR values assigned in dict key order:")
    eir_index = 0
    for product in eir_dict.keys():
        eir_values = eir_dict[product]
        for j, eir in enumerate(eir_values):
            print(f"     {eir_index}: {product} -> {eir:.4f}")
            eir_index += 1

    print("   Lifetime values assigned in dict key order:")
    lifetime_index = 0
    for product in lifetime_dict.keys():
        lifetime_values = lifetime_dict[product]
        for j, lifetime in enumerate(lifetime_values):
            print(f"     {lifetime_index}: {product} -> {lifetime}")
            lifetime_index += 1

def debug_correct_approach():
    """Show the correct approach to maintain order."""
    print("\n=== Correct Approach Debug ===")

    from synthetic_data.eir_generator import generate_eir_for_multiple_products
    from synthetic_data.lifetime_generator import generate_lifetime_for_multiple_products

    product_assignments = ["credit_card", "auto_loan", "mortgage", "credit_card", "auto_loan"]

    print("1. Correct EIR generation (maintaining order):")
    eir_dict = generate_eir_for_multiple_products(
        product_assignments=product_assignments,
        seed=42
    )

    # Create index mapping for each product
    product_indices = {}
    for i, product in enumerate(product_assignments):
        if product not in product_indices:
            product_indices[product] = []
        product_indices[product].append(i)

    # Assign EIR values in the correct order
    all_eirs_correct = [None] * len(product_assignments)
    for product, eir_values in eir_dict.items():
        indices = product_indices[product]
        for i, idx in enumerate(indices):
            all_eirs_correct[idx] = eir_values[i]

    print(f"   Correct EIR values: {[f'{eir:.4f}' for eir in all_eirs_correct]}")

    print("\n2. Correct lifetime generation (maintaining order):")
    lifetime_dict = generate_lifetime_for_multiple_products(
        product_assignments=product_assignments,
        seed=42
    )

    # Assign lifetime values in the correct order
    all_lifetimes_correct = [None] * len(product_assignments)
    for product, lifetime_values in lifetime_dict.items():
        indices = product_indices[product]
        for i, idx in enumerate(indices):
            all_lifetimes_correct[idx] = lifetime_values[i]

    print(f"   Correct lifetime values: {all_lifetimes_correct}")

def main():
    print("Order Mismatch Debug Analysis")
    print("=" * 50)

    debug_order_mismatch()
    debug_correct_approach()

    print("\n" + "=" * 50)
    print("Order mismatch analysis complete!")

if __name__ == "__main__":
    main()