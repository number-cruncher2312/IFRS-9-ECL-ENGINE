#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synthetic_data.portfolio_integrator import generate_phase1_portfolio, generate_portfolio_statistics

def main():
    print("Testing portfolio integrator...")

    try:
        # Generate a small test portfolio
        portfolio = generate_phase1_portfolio(master_seed=42, n_loans=100)
        print(f"Portfolio generated successfully: {len(portfolio)} loans")

        # Generate statistics
        stats = generate_portfolio_statistics(portfolio)
        print(f"Portfolio statistics generated")

        # Check for EIR out-of-bounds issues
        print("\nEIR validation:")
        for product in portfolio['product_type'].unique():
            product_data = portfolio[portfolio['product_type'] == product]
            eir_values = product_data['eir']
            from synthetic_data.product_taxonomy import PRODUCT_TAXONOMY
            config = PRODUCT_TAXONOMY[product]['eir']
            min_eir, max_eir = config['min'], config['max']

            out_of_bounds = (eir_values < min_eir) | (eir_values > max_eir)
            if np.any(out_of_bounds):
                print(f"  {product}: {np.sum(out_of_bounds)} EIR values out of bounds [{min_eir}, {max_eir}]")
                print(f"    Min: {eir_values.min():.4f}, Max: {eir_values.max():.4f}")
            else:
                print(f"  {product}: All EIR values within bounds [{min_eir}, {max_eir}]")

        # Check for lifetime out-of-bounds issues
        print("\nLifetime validation:")
        for product in portfolio['product_type'].unique():
            product_data = portfolio[portfolio['product_type'] == product]
            lifetime_values = product_data['remaining_lifetime_months']
            config = PRODUCT_TAXONOMY[product]['remaining_lifetime_months']
            min_lifetime, max_lifetime = config['min'], config['max']

            out_of_bounds = (lifetime_values < min_lifetime) | (lifetime_values > max_lifetime)
            if np.any(out_of_bounds):
                print(f"  {product}: {np.sum(out_of_bounds)} lifetime values out of bounds [{min_lifetime}, {max_lifetime}]")
                print(f"    Min: {lifetime_values.min():.1f}, Max: {lifetime_values.max():.1f}")
            else:
                print(f"  {product}: All lifetime values within bounds [{min_lifetime}, {max_lifetime}]")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import numpy as np
    main()