"""
Balance Generator Module for Synthetic Loan Dataset
==================================================

This module implements positive, right-skewed balance generation using a bounded/truncated
lognormal approach based on the product taxonomy min, median, and max values.

Key Features:
- Uses bounded/truncated lognormal distribution for right-skewed balance generation
- Respects product taxonomy min, median, and max values as sampling bounds
- Generates positive balances only
- For V1: EAD = balance (no unused-limit or CCF model)
- Explicit seeds for reproducibility
- Product differentiation through taxonomy-based parameters

Methodology:
-----------
1. For each product, extract min, median, max balance values from taxonomy
2. Fit a lognormal distribution to these bounds using:
   - mu: location parameter derived from median
   - sigma: scale parameter calibrated to span min-max range
3. Apply truncation to ensure all generated values stay within [min, max] bounds
4. Generate right-skewed, positive balances that respect product characteristics
5. Set EAD = balance for all products (V1 simplification)

Important Notes:
- Does NOT use PD as input (as specified)
- Does NOT implement borrower-specific balance adjustments (V1 baseline)
- Taxonomy values are treated as synthetic V1 assumptions, not empirical calibration
- All generated balances are in USD
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import lognorm, truncnorm
from .product_taxonomy import PRODUCT_TAXONOMY, PRODUCT_TYPES

def _calculate_lognormal_params(min_val: float, median_val: float, max_val: float) -> Tuple[float, float]:
    """
    Calculate lognormal distribution parameters (mu, sigma) that fit within given bounds.

    Args:
        min_val: Minimum bound (truncation lower limit)
        median_val: Median value (used to set mu parameter)
        max_val: Maximum bound (truncation upper limit)

    Returns:
        Tuple of (mu, sigma) parameters for lognormal distribution
    """
    # For lognormal, median = exp(mu), so mu = log(median)
    mu = np.log(median_val)

    # Calculate sigma to span the range from min to max
    # We want the distribution to cover the min-max range with reasonable probability
    # Use the relationship: max_val ≈ exp(mu + k*sigma), min_val ≈ exp(mu - k*sigma)
    # For k=3 (3-sigma range covers ~99.7% of distribution), we can solve for sigma

    # Calculate sigma based on the range needed to cover min to max
    # We use a conservative approach to ensure we can truncate properly
    log_min = np.log(min_val)
    log_max = np.log(max_val)

    # Target 99% of the distribution to be within bounds
    # Using 2.576 sigma for 99% confidence interval (two-tailed)
    target_sigma = (log_max - log_min) / (2 * 2.576)

    # Ensure sigma is positive and reasonable
    sigma = max(target_sigma, 0.1)  # Minimum sigma to avoid degenerate distributions

    return mu, sigma

def _generate_truncated_lognormal_samples(
    min_val: float,
    median_val: float,
    max_val: float,
    size: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate truncated lognormal samples within specified bounds.

    Args:
        min_val: Minimum bound (inclusive)
        median_val: Median value for distribution fitting
        max_val: Maximum bound (inclusive)
        size: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Array of generated balance values
    """
    # Calculate lognormal parameters
    mu, sigma = _calculate_lognormal_params(min_val, median_val, max_val)

    # For truncated lognormal, we work in log-space for the truncation
    log_min = np.log(min_val)
    log_max = np.log(max_val)

    # Calculate truncation bounds in standard normal space
    # For lognormal, we truncate the underlying normal distribution
    a = (log_min - mu) / sigma
    b = (log_max - mu) / sigma

    # Generate truncated normal samples in log-space
    rng = np.random.default_rng(seed)
    truncated_normal_samples = truncnorm.rvs(
        a=a, b=b, loc=mu, scale=sigma, size=size, random_state=rng
    )

    # Convert back to original scale (exponentiate)
    samples = np.exp(truncated_normal_samples)

    # Ensure all values are within bounds (floating point safety)
    samples = np.clip(samples, min_val, max_val)

    return samples

def generate_balances_for_product(
    product_type: str,
    size: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate right-skewed balances for a specific product type.

    Args:
        product_type: Product type from PRODUCT_TYPES
        size: Number of balance samples to generate
        seed: Random seed for reproducibility

    Returns:
        Array of generated balance values

    Raises:
        ValueError: If product_type is unknown or balance config is invalid
    """
    # Validate product type
    if product_type not in PRODUCT_TAXONOMY:
        raise ValueError(
            f"Unknown product_type: {product_type!r}. "
            f"Valid types: {sorted(PRODUCT_TYPES)}"
        )

    # Get balance configuration from taxonomy
    balance_config = PRODUCT_TAXONOMY[product_type]["balance"]

    # Extract min, median, max values
    min_val = balance_config["min"]
    median_val = balance_config["median"]
    max_val = balance_config["max"]

    # Validate balance configuration
    if not (min_val > 0 and median_val > 0 and max_val > 0):
        raise ValueError(
            f"Balance values for {product_type} must be positive. "
            f"Got min={min_val}, median={median_val}, max={max_val}"
        )

    if not (min_val <= median_val <= max_val):
        raise ValueError(
            f"Balance values for {product_type} must satisfy min <= median <= max. "
            f"Got min={min_val}, median={median_val}, max={max_val}"
        )

    # Generate truncated lognormal samples
    balances = _generate_truncated_lognormal_samples(
        min_val, median_val, max_val, size, seed
    )

    return balances

def generate_balances_for_multiple_products(
    product_assignments: List[str],
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate balances for multiple products based on product assignments.

    Args:
        product_assignments: List of product types (one per loan)
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping product_type -> array of balances

    Raises:
        ValueError: If product_assignments is empty or contains invalid products
    """
    if not product_assignments:
        raise ValueError("product_assignments cannot be empty")

    if len(product_assignments) != len([p for p in product_assignments if p in PRODUCT_TYPES]):
        invalid_products = set(product_assignments) - PRODUCT_TYPES
        raise ValueError(f"Invalid product types in assignments: {invalid_products}")

    # Group assignments by product type
    product_groups = {}
    for product in product_assignments:
        if product not in product_groups:
            product_groups[product] = []
        product_groups[product].append(product)

    # Generate balances for each product group
    result = {}
    for product, assignments in product_groups.items():
        size = len(assignments)
        # Use deterministic seed based on main seed and product for reproducibility
        product_seed = None if seed is None else hash((seed, product)) % 1000000
        balances = generate_balances_for_product(product, size, product_seed)
        result[product] = balances

    return result

def generate_ead_from_balances(balances: np.ndarray) -> np.ndarray:
    """
    Generate EAD values from balances (V1: EAD = balance).

    Args:
        balances: Array of balance values

    Returns:
        Array of EAD values (identical to balances for V1)
    """
    return balances.copy()

def validate_balance_generation(
    balances: Dict[str, np.ndarray],
    product_assignments: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Validate generated balances and return statistics.

    Args:
        balances: Dictionary mapping product_type -> array of balances
        product_assignments: List of product assignments

    Returns:
        Dictionary of statistics by product

    Raises:
        ValueError: If validation fails
    """
    # Check that all products in balances are in assignments
    balance_products = set(balances.keys())
    assignment_products = set(product_assignments)

    if not balance_products.issubset(assignment_products):
        raise ValueError("Balances contain products not in assignments")

    # Check that total count matches
    total_balances = sum(len(b) for b in balances.values())
    if total_balances != len(product_assignments):
        raise ValueError(f"Total balance count {total_balances} != assignment count {len(product_assignments)}")

    # Validate each product's balances
    stats = {}
    for product, product_balances in balances.items():
        # Check positivity
        if not np.all(product_balances > 0):
            raise ValueError(f"Product {product} has non-positive balances")

        # Check bounds
        config = PRODUCT_TAXONOMY[product]["balance"]
        min_val = config["min"]
        max_val = config["max"]

        if not np.all(product_balances >= min_val):
            raise ValueError(f"Product {product} has balances below minimum {min_val}")

        if not np.all(product_balances <= max_val):
            raise ValueError(f"Product {product} has balances above maximum {max_val}")

        # Calculate statistics
        stats[product] = {
            "count": len(product_balances),
            "min": float(np.min(product_balances)),
            "median": float(np.median(product_balances)),
            "mean": float(np.mean(product_balances)),
            "max": float(np.max(product_balances)),
            "std": float(np.std(product_balances)),
            "skewness": float(pd.Series(product_balances).skew())
        }

    return stats

# Internal validation tests
def _run_validation_tests() -> None:
    """Run internal validation tests for the balance generator."""
    print("Running balance generator validation tests...")

    # Test 1: Parameter calculation
    mu, sigma = _calculate_lognormal_params(1000, 5000, 20000)
    assert mu > 0 and sigma > 0, "Lognormal parameters should be positive"

    # Test 2: Sample generation
    samples = _generate_truncated_lognormal_samples(1000, 5000, 20000, 100, seed=42)
    assert len(samples) == 100, "Should generate correct number of samples"
    assert np.all(samples >= 1000) and np.all(samples <= 20000), "Samples should be within bounds"
    assert np.all(samples > 0), "All samples should be positive"

    # Test 3: Product balance generation
    for product in ["mortgage", "auto_loan", "credit_card"]:
        balances = generate_balances_for_product(product, 50, seed=42)
        assert len(balances) == 50, f"Should generate 50 balances for {product}"
        config = PRODUCT_TAXONOMY[product]["balance"]
        assert np.all(balances >= config["min"]), f"{product} balances should respect min"
        assert np.all(balances <= config["max"]), f"{product} balances should respect max"

    # Test 4: EAD generation
    test_balances = np.array([1000, 2000, 3000])
    ead_values = generate_ead_from_balances(test_balances)
    assert np.array_equal(ead_values, test_balances), "EAD should equal balances for V1"

    # Test 5: Multiple product generation
    assignments = ["mortgage", "auto_loan", "credit_card", "mortgage"]
    balances_dict = generate_balances_for_multiple_products(assignments, seed=42)
    assert set(balances_dict.keys()) == {"mortgage", "auto_loan", "credit_card"}
    assert len(balances_dict["mortgage"]) == 2
    assert len(balances_dict["auto_loan"]) == 1
    assert len(balances_dict["credit_card"]) == 1

    # Test 6: Validation
    stats = validate_balance_generation(balances_dict, assignments)
    assert "mortgage" in stats and "auto_loan" in stats and "credit_card" in stats

    print("Balance generator validation: OK")

if __name__ == "__main__":
    _run_validation_tests()
    print("Balance generator module initialized successfully.")