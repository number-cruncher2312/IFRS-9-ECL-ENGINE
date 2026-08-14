#!/usr/bin/env python3

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_eir_generation():
    """Debug EIR generation to understand the bounds issue."""
    print("=== EIR Generation Debug ===")

    from synthetic_data.eir_generator import generate_eir_for_product, generate_eir_for_multiple_products
    from synthetic_data.product_taxonomy import PRODUCT_TAXONOMY

    # Test standalone EIR generation
    print("1. Testing standalone EIR generation:")
    for product in ["credit_card", "auto_loan", "mortgage"]:
        eirs = generate_eir_for_product(product, n_loans=10, seed=42)
        config = PRODUCT_TAXONOMY[product]['eir']
        min_eir, max_eir = config['min'], config['max']

        out_of_bounds = (np.array(eirs) < min_eir) | (np.array(eirs) > max_eir)
        print(f"  {product}: {sum(out_of_bounds)}/{len(eirs)} out of bounds [{min_eir}, {max_eir}]")
        print(f"    Values: {[f'{eir:.4f}' for eir in eirs[:5]]}...")

    # Test multiple products generation
    print("\n2. Testing multiple products EIR generation:")
    product_assignments = ["credit_card"] * 10 + ["auto_loan"] * 10 + ["mortgage"] * 10
    eir_dict = generate_eir_for_multiple_products(product_assignments, seed=42)

    for product, eirs in eir_dict.items():
        config = PRODUCT_TAXONOMY[product]['eir']
        min_eir, max_eir = config['min'], config['max']

        out_of_bounds = (np.array(eirs) < min_eir) | (np.array(eirs) > max_eir)
        print(f"  {product}: {sum(out_of_bounds)}/{len(eirs)} out of bounds [{min_eir}, {max_eir}]")
        print(f"    Values: {[f'{eir:.4f}' for eir in eirs[:5]]}...")

def debug_lifetime_generation():
    """Debug lifetime generation to understand the bounds issue."""
    print("\n=== Lifetime Generation Debug ===")

    from synthetic_data.lifetime_generator import generate_lifetime_for_product, generate_lifetime_for_multiple_products
    from synthetic_data.product_taxonomy import PRODUCT_TAXONOMY

    # Test standalone lifetime generation
    print("1. Testing standalone lifetime generation:")
    for product in ["credit_card", "auto_loan", "mortgage"]:
        lifetimes = generate_lifetime_for_product(product, size=10, seed=42)
        config = PRODUCT_TAXONOMY[product]['remaining_lifetime_months']
        min_lifetime, max_lifetime = config['min'], config['max']

        out_of_bounds = (lifetimes < min_lifetime) | (lifetimes > max_lifetime)
        print(f"  {product}: {sum(out_of_bounds)}/{len(lifetimes)} out of bounds [{min_lifetime}, {max_lifetime}]")
        print(f"    Values: {lifetimes[:5].tolist()}...")

    # Test multiple products generation
    print("\n2. Testing multiple products lifetime generation:")
    product_assignments = ["credit_card"] * 10 + ["auto_loan"] * 10 + ["mortgage"] * 10
    lifetime_dict = generate_lifetime_for_multiple_products(product_assignments, seed=42)

    for product, lifetimes in lifetime_dict.items():
        config = PRODUCT_TAXONOMY[product]['remaining_lifetime_months']
        min_lifetime, max_lifetime = config['min'], config['max']

        out_of_bounds = (lifetimes < min_lifetime) | (lifetimes > max_lifetime)
        print(f"  {product}: {sum(out_of_bounds)}/{len(lifetimes)} out of bounds [{min_lifetime}, {max_lifetime}]")
        print(f"    Values: {lifetimes[:5].tolist()}...")

def debug_integrator_calls():
    """Debug how the integrator calls the generators."""
    print("\n=== Integrator Call Debug ===")

    from synthetic_data.eir_generator import generate_eir_for_multiple_products
    from synthetic_data.lifetime_generator import generate_lifetime_for_multiple_products
    from synthetic_data.product_taxonomy import PRODUCT_TAXONOMY

    # Simulate the exact call pattern used by the integrator
    product_assignments = ["credit_card"] * 30 + ["auto_loan"] * 20 + ["mortgage"] * 50

    print("1. Testing integrator-style EIR generation:")
    # This is exactly how the integrator calls it
    eir_dict = generate_eir_for_multiple_products(
        product_assignments=product_assignments,
        seed=42  # This would be the component seed in the integrator
    )

    all_eirs = []
    for product in eir_dict.keys():
        eir_values = eir_dict[product]
        all_eirs.extend(eir_values)

    print(f"  Generated {len(all_eirs)} EIR values")
    print(f"  Products: {list(eir_dict.keys())}")
    print(f"  Counts: {[len(eirs) for eirs in eir_dict.values()]}")

    # Check bounds for each product
    for product, eirs in eir_dict.items():
        config = PRODUCT_TAXONOMY[product]['eir']
        min_eir, max_eir = config['min'], config['max']
        eir_array = np.array(eirs)

        out_of_bounds = (eir_array < min_eir) | (eir_array > max_eir)
        if np.any(out_of_bounds):
            print(f"  {product}: {np.sum(out_of_bounds)} EIR values out of bounds [{min_eir}, {max_eir}]")
            print(f"    Min: {eir_array.min():.4f}, Max: {eir_array.max():.4f}")
            print(f"    Problematic: {eir_array[out_of_bounds][:3].tolist()}")
        else:
            print(f"  {product}: All EIR values within bounds [{min_eir}, {max_eir}]")

    print("\n2. Testing integrator-style lifetime generation:")
    lifetime_dict = generate_lifetime_for_multiple_products(
        product_assignments=product_assignments,
        seed=42  # This would be the component seed in the integrator
    )

    all_lifetimes = []
    for product in lifetime_dict.keys():
        lifetime_values = lifetime_dict[product]
        all_lifetimes.extend(lifetime_values)

    print(f"  Generated {len(all_lifetimes)} lifetime values")
    print(f"  Products: {list(lifetime_dict.keys())}")
    print(f"  Counts: {[len(lifetimes) for lifetimes in lifetime_dict.values()]}")

    # Check bounds for each product
    for product, lifetimes in lifetime_dict.items():
        config = PRODUCT_TAXONOMY[product]['remaining_lifetime_months']
        min_lifetime, max_lifetime = config['min'], config['max']
        lifetime_array = np.array(lifetimes)

        out_of_bounds = (lifetime_array < min_lifetime) | (lifetime_array > max_lifetime)
        if np.any(out_of_bounds):
            print(f"  {product}: {np.sum(out_of_bounds)} lifetime values out of bounds [{min_lifetime}, {max_lifetime}]")
            print(f"    Min: {lifetime_array.min():.1f}, Max: {lifetime_array.max():.1f}")
            print(f"    Problematic: {lifetime_array[out_of_bounds][:3].tolist()}")
        else:
            print(f"  {product}: All lifetime values within bounds [{min_lifetime}, {max_lifetime}]")

def main():
    print("Integration Debug Analysis")
    print("=" * 50)

    debug_eir_generation()
    debug_lifetime_generation()
    debug_integrator_calls()

    print("\n" + "=" * 50)
    print("Debug analysis complete!")

if __name__ == "__main__":
    main()