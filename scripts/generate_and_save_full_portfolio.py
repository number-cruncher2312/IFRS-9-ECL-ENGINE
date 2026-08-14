"""
Generate and save the integrated V1 Phase 1 IFRS 9 portfolio.
"""
import numpy

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.portfolio_integrator import generate_phase1_portfolio


def generate_and_save_full_portfolio(
    output_filename: str = "outputs/full_10000_loan_portfolio.csv",
    seed: int = 42,
) -> None:
    """Generate the full 10,000-loan integrated portfolio and save it."""
    print("Generating integrated 10,000-loan V1 Phase 1 portfolio...")
    print("=" * 60)

    portfolio = generate_phase1_portfolio(
        master_seed=seed,
        n_loans=10000,
    )

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print(f"Saving portfolio to {output_filename}...")
    portfolio.to_csv(output_filename, index=False)

    print("\n" + "=" * 60)
    print("INTEGRATED PORTFOLIO SUMMARY")
    print("=" * 60)

    print(f"Total loans: {len(portfolio):,}")
    print(f"Total EAD: ${portfolio['ead'].sum():,.2f}")
    print(f"Mean PD_current: {portfolio['pd_current'].mean():.6f}")
    print(f"Expected defaults: {portfolio['pd_current'].sum():,.2f}")
    print(f"Actual defaults: {portfolio['default_status'].sum():,}")
    print(f"Default rate: {portfolio['default_status'].mean():.4%}")

    print("\nProduct distribution:")
    print(portfolio["product_type"].value_counts())

    print("\nColumns:")
    print(list(portfolio.columns))

    print("\n" + "=" * 60)
    print("PORTFOLIO GENERATION AND SAVE COMPLETE")
    print("=" * 60)
    print(f"Saved to: {output_filename}")


if __name__ == "__main__":
    generate_and_save_full_portfolio()