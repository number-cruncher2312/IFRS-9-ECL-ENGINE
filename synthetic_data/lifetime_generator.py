"""
Remaining Lifetime Generator Module for Synthetic Loan Dataset
=============================================================

V1 Remaining Lifetime Generation Component for Synthetic Loan Dataset

This module implements the V1 methodology for generating synthetic remaining lifetime
values that represent the contractual remaining tenor of loans at the snapshot date.

Key Features:
- Generates remaining lifetime values based on product taxonomy configuration
- Uses product type as primary determinant of lifetime distribution
- Applies bounded distribution centered around configured median
- Ensures lifetime values are positive integers within product-specific min/max bounds
- Supports explicit seeding for full reproducibility
- Does NOT use PD, default status, EIR, LGD, EAD, or any risk metrics
- Does NOT implement borrower-specific adjustments

Methodology:
Remaining Lifetime = f(product_type, product-specific_distribution)

where:
- product_type determines the distribution parameters (min, median, max)
- distribution is bounded and centered around the median
- values are generated as integers representing months
- all values remain within configured min/max bounds

V1 Remaining Lifetime Configuration (from product_taxonomy.py):
- mortgage: min 60 months, median 180 months, max 360 months
- auto_loan: min 12 months, median 48 months, max 72 months
- credit_card: min 1 month, median 18 months, max 60 months
- student_loan: min 12 months, median 120 months, max 240 months (inactive V1)
- other_personal_loan: min 6 months, median 36 months, max 84 months (inactive V1)

Important Constraints:
- Remaining lifetime represents CONTRACTUAL tenor, not risk-driven maturity
- Must NOT use PD, default status, EIR, LGD, EAD, or any risk metrics
- Must remain within configured min/max bounds for each product
- Must generate positive integer values only
- Should be reasonably centered around configured median
- Must be reproducible with explicit seed
- Product type is the PRIMARY determinant of lifetime distribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.stats import truncnorm
from scipy.stats import beta as beta_dist
from .product_taxonomy import PRODUCT_TAXONOMY, PRODUCT_TYPES

def _calculate_beta_params(min_val: int, median_val: int, max_val: int) -> Tuple[float, float]:
    """
    Calculate beta distribution parameters (alpha, beta) that fit within given bounds
    and are centered around the median.

    Args:
        min_val: Minimum bound (inclusive)
        median_val: Median value for distribution
        max_val: Maximum bound (inclusive)

    Returns:
        Tuple of (alpha, beta) parameters for beta distribution
    """
    # Transform to [0,1] range for beta distribution
    # Beta distribution is defined on [0,1], so we need to scale our values
    scaled_min = 0.0
    scaled_max = 1.0
    scaled_median = (median_val - min_val) / (max_val - min_val)

    # For beta distribution, we want to find alpha and beta such that:
    # median ≈ alpha / (alpha + beta)
    # We'll use a simple approach: set alpha = beta * (median / (1 - median))

    if scaled_median <= 0 or scaled_median >= 1:
        raise ValueError(f"Median {median_val} must be between min {min_val} and max {max_val}")

    # Use a reasonable spread - we want some variation but centered around median
    # A common approach is to set the mode near the median
    # For simplicity, we'll use alpha = beta * (median / (1 - median))
    # and choose beta to control the spread

    # Calculate alpha and beta to center around median
    # We use a concentration parameter to control how peaked the distribution is
    concentration = 4.0  # Higher = more concentrated around median

    if scaled_median < 0.5:
        # Left-skewed case (median < 0.5)
        alpha = concentration * scaled_median
        beta = concentration * (1 - scaled_median)
    else:
        # Right-skewed or symmetric case (median >= 0.5)
        alpha = concentration * scaled_median
        beta = concentration * (1 - scaled_median)

    # Ensure parameters are reasonable
    alpha = max(alpha, 1.0)
    beta = max(beta, 1.0)

    return alpha, beta

def _generate_bounded_beta_samples(
    min_val: int,
    median_val: int,
    max_val: int,
    size: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate bounded beta distribution samples within specified bounds.

    Args:
        min_val: Minimum bound (inclusive)
        median_val: Median value for distribution
        max_val: Maximum bound (inclusive)
        size: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Array of generated lifetime values (as integers)
    """
    # Calculate beta parameters
    alpha, beta = _calculate_beta_params(min_val, median_val, max_val)

    # Generate beta samples in [0,1] range
    rng = np.random.default_rng(seed)
    beta_samples = beta_dist.rvs(alpha, beta, size=size, random_state=rng)

    # Scale to [min_val, max_val] range
    scaled_samples = min_val + beta_samples * (max_val - min_val)

    # Round to nearest integer and ensure bounds
    int_samples = np.round(scaled_samples).astype(int)
    int_samples = np.clip(int_samples, min_val, max_val)

    return int_samples

def generate_lifetime_for_product(
    product_type: str,
    size: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate remaining lifetime values for a specific product type.

    Args:
        product_type: Product type from PRODUCT_TYPES
        size: Number of lifetime samples to generate
        seed: Random seed for reproducibility

    Returns:
        Array of generated lifetime values (integers, months)

    Raises:
        ValueError: If product_type is unknown or parameters are invalid
    """
    # Validate product type
    if product_type not in PRODUCT_TAXONOMY:
        raise ValueError(
            f"Unknown product_type: {product_type!r}. "
            f"Valid types: {sorted(PRODUCT_TYPES)}"
        )

    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    # Get lifetime configuration from taxonomy
    lifetime_config = PRODUCT_TAXONOMY[product_type]["remaining_lifetime_months"]

    # Extract min, median, max values
    min_val = lifetime_config["min"]
    median_val = lifetime_config["median"]
    max_val = lifetime_config["max"]

    # Validate lifetime configuration
    if not (min_val > 0 and median_val > 0 and max_val > 0):
        raise ValueError(
            f"Lifetime values for {product_type} must be positive. "
            f"Got min={min_val}, median={median_val}, max={max_val}"
        )

    if not (min_val <= median_val <= max_val):
        raise ValueError(
            f"Lifetime values for {product_type} must satisfy min <= median <= max. "
            f"Got min={min_val}, median={median_val}, max={max_val}"
        )

    # Generate bounded beta samples
    lifetimes = _generate_bounded_beta_samples(
        min_val, median_val, max_val, size, seed
    )

    return lifetimes

def generate_lifetime_for_multiple_products(
    product_assignments: List[str],
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate lifetime values for multiple products based on product assignments.

    Args:
        product_assignments: List of product types (one per loan)
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping product_type -> array of lifetime values

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

    # Generate lifetimes for each product group
    result = {}
    for product, assignments in product_groups.items():
        size = len(assignments)
        # Use deterministic seed based on main seed and product for reproducibility
        product_seed = None if seed is None else hash((seed, product)) % 1000000
        lifetimes = generate_lifetime_for_product(product, size, product_seed)
        result[product] = lifetimes

    return result

def generate_lifetime_dataframe(
    product_assignments: List[str],
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate lifetime values and return as DataFrame with product assignments.

    Args:
        product_assignments: List of product types (one per loan)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: product_type, remaining_lifetime_months

    Raises:
        ValueError: If product_assignments is empty or contains invalid products
    """
    if not product_assignments:
        raise ValueError("product_assignments cannot be empty")

    # Generate lifetime values
    lifetime_dict = generate_lifetime_for_multiple_products(
        product_assignments=product_assignments,
        seed=seed
    )

    # Create DataFrame
    all_products = []
    all_lifetimes = []

    for product, lifetime_values in lifetime_dict.items():
        all_products.extend([product] * len(lifetime_values))
        all_lifetimes.extend(lifetime_values)

    # Verify we have the same number of lifetime values as product assignments
    if len(all_products) != len(product_assignments):
        raise RuntimeError(
            f"Lifetime generation error: expected {len(product_assignments)} values, "
            f"got {len(all_products)}"
        )

    return pd.DataFrame({
        "product_type": all_products,
        "remaining_lifetime_months": all_lifetimes
    })

def validate_lifetime_values(
    lifetime_values: Union[List[int], np.ndarray],
    product_type: str
) -> Dict[str, bool]:
    """
    Validate that lifetime values conform to product constraints.

    Args:
        lifetime_values: List or array of lifetime values to validate
        product_type: Product type for validation

    Returns:
        Dictionary of validation results with boolean flags
    """
    # Get lifetime configuration for the product
    lifetime_config = PRODUCT_TAXONOMY[product_type]["remaining_lifetime_months"]
    min_val = lifetime_config["min"]
    max_val = lifetime_config["max"]

    validation_results = {}

    # Convert to numpy array for easier validation
    lifetime_array = np.array(lifetime_values)

    # Check all values are integers
    validation_results["all_integers"] = np.all(lifetime_array == np.round(lifetime_array))

    # Check all values are within bounds
    validation_results["all_within_bounds"] = (
        (lifetime_array >= min_val).all() and
        (lifetime_array <= max_val).all()
    )

    # Check all values are positive
    validation_results["all_positive"] = (lifetime_array > 0).all()

    # Check reasonable centering around median
    median_val = lifetime_config["median"]
    mean_lifetime = np.mean(lifetime_array)
    validation_results["reasonably_centered"] = (
        abs(mean_lifetime - median_val) <= (max_val - min_val) * 0.3
    )

    return validation_results

def get_lifetime_statistics(
    lifetime_values: Union[List[int], np.ndarray, pd.Series],
    product_type: str
) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for lifetime values.

    Args:
        lifetime_values: Lifetime values to analyze
        product_type: Product type for context

    Returns:
        Dictionary of statistical measures
    """
    lifetime_array = np.array(lifetime_values)
    lifetime_config = PRODUCT_TAXONOMY[product_type]["remaining_lifetime_months"]
    median_config = lifetime_config["median"]

    return {
        "count": len(lifetime_array),
        "min": float(np.min(lifetime_array)),
        "median": float(np.median(lifetime_array)),
        "mean": float(np.mean(lifetime_array)),
        "max": float(np.max(lifetime_array)),
        "std": float(np.std(lifetime_array)),
        "median_config": float(median_config),
        "distance_from_median": float(np.mean(np.abs(lifetime_array - median_config))),
        "coefficient_of_variation": float(np.std(lifetime_array) / np.mean(lifetime_array)) if np.mean(lifetime_array) > 0 else 0.0
    }

def validate_lifetime_reproducibility(
    product_assignments: List[str],
    seed: int,
    n_trials: int = 3
) -> bool:
    """
    Validate that lifetime generation is reproducible with the same seed.

    Args:
        product_assignments: List of product types
        seed: Seed to use for reproducibility test
        n_trials: Number of trials to run

    Returns:
        True if all trials produce identical results, False otherwise
    """
    # Generate multiple lifetime dataframes with the same seed
    lifetime_dfs = []
    for i in range(n_trials):
        lifetime_df = generate_lifetime_dataframe(
            product_assignments=product_assignments,
            seed=seed
        )
        lifetime_dfs.append(lifetime_df)

    # Check that all dataframes are identical
    reference = lifetime_dfs[0]
    for i, lifetime_df in enumerate(lifetime_dfs[1:], 1):
        if not lifetime_df.equals(reference):
            print(f"Reproducibility check failed: trial {i} differs from reference")
            return False

    print(f"Lifetime reproducibility validated: {n_trials} trials with seed {seed} produced identical results")
    return True

def generate_lifetime_report(
    product_assignments: List[str],
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive lifetime report with statistics and validation.

    Args:
        product_assignments: List of product types
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing lifetime data, statistics, and validation results
    """
    # Generate lifetime values
    lifetime_df = generate_lifetime_dataframe(
        product_assignments=product_assignments,
        seed=seed
    )

    # Calculate statistics by product
    statistics = {}
    validation = {}

    for product in lifetime_df["product_type"].unique():
        product_data = lifetime_df[lifetime_df["product_type"] == product]
        product_lifetimes = product_data["remaining_lifetime_months"].values

        # Calculate statistics
        statistics[product] = get_lifetime_statistics(product_lifetimes, product)

        # Validate values
        validation[product] = validate_lifetime_values(product_lifetimes, product)

    # Overall validation
    overall_validation = {
        "all_products_valid": all(
            stats["all_within_bounds"] and stats["all_positive"] and stats["all_integers"]
            for stats in validation.values()
        ),
        "product_count": len(statistics),
        "total_loans": len(lifetime_df)
    }

    return {
        "lifetime_data": lifetime_df,
        "statistics": statistics,
        "validation": validation,
        "overall_validation": overall_validation,
        "metadata": {
            "seed": seed,
            "generation_timestamp": pd.Timestamp.now(),
            "product_distribution": lifetime_df["product_type"].value_counts(normalize=True).to_dict()
        }
    }

# Internal validation tests
def _run_validation_tests() -> None:
    """Run internal validation tests for the lifetime generator."""
    print("Running lifetime generator validation tests...")

    # Test 1: Parameter calculation
    alpha, beta = _calculate_beta_params(12, 48, 72)
    assert alpha > 0 and beta > 0, "Beta parameters should be positive"

    # Test 2: Sample generation
    samples = _generate_bounded_beta_samples(12, 48, 72, 100, seed=42)
    assert len(samples) == 100, "Should generate correct number of samples"
    assert np.all(samples >= 12) and np.all(samples <= 72), "Samples should be within bounds"
    assert np.all(samples > 0), "All samples should be positive"
    assert np.all(samples == np.round(samples)), "All samples should be integers"

    # Test 3: Product lifetime generation
    for product in ["mortgage", "auto_loan", "credit_card"]:
        lifetimes = generate_lifetime_for_product(product, 50, seed=42)
        assert len(lifetimes) == 50, f"Should generate 50 lifetimes for {product}"
        config = PRODUCT_TAXONOMY[product]["remaining_lifetime_months"]
        assert np.all(lifetimes >= config["min"]), f"{product} lifetimes should respect min"
        assert np.all(lifetimes <= config["max"]), f"{product} lifetimes should respect max"
        assert np.all(lifetimes > 0), f"{product} lifetimes should be positive"
        assert np.all(lifetimes == np.round(lifetimes)), f"{product} lifetimes should be integers"

    # Test 4: Multiple product generation
    assignments = ["mortgage", "auto_loan", "credit_card", "mortgage"]
    lifetimes_dict = generate_lifetime_for_multiple_products(assignments, seed=42)
    assert set(lifetimes_dict.keys()) == {"mortgage", "auto_loan", "credit_card"}
    assert len(lifetimes_dict["mortgage"]) == 2
    assert len(lifetimes_dict["auto_loan"]) == 1
    assert len(lifetimes_dict["credit_card"]) == 1

    # Test 5: DataFrame generation
    lifetime_df = generate_lifetime_dataframe(assignments, seed=42)
    assert len(lifetime_df) == len(assignments), "DataFrame should have same length as assignments"
    assert "product_type" in lifetime_df.columns, "Should have product_type column"
    assert "remaining_lifetime_months" in lifetime_df.columns, "Should have remaining_lifetime_months column"

    # Test 6: Validation
    validation = validate_lifetime_values(lifetimes_dict["auto_loan"], "auto_loan")
    assert validation["all_within_bounds"], "All lifetimes should be within bounds"
    assert validation["all_positive"], "All lifetimes should be positive"
    assert validation["all_integers"], "All lifetimes should be integers"

    # Test 7: Statistics
    stats = get_lifetime_statistics(lifetimes_dict["auto_loan"], "auto_loan")
    assert "mean" in stats, "Statistics should include mean"
    assert "std" in stats, "Statistics should include std"
    assert "median_config" in stats, "Statistics should include median_config"

    # Test 8: Reproducibility
    assert validate_lifetime_reproducibility(["mortgage", "auto_loan"], seed=123, n_trials=2), "Should be reproducible"

    # Test 9: Error handling
    try:
        generate_lifetime_for_product("invalid_product", 10)
        assert False, "Should raise error for invalid product"
    except ValueError:
        pass  # Expected

    try:
        generate_lifetime_for_product("mortgage", 0)
        assert False, "Should raise error for size=0"
    except ValueError:
        pass  # Expected

    # Test 10: Report generation
    report = generate_lifetime_report(assignments, seed=42)
    assert "lifetime_data" in report, "Report should contain lifetime_data"
    assert "statistics" in report, "Report should contain statistics"
    assert "validation" in report, "Report should contain validation"

    print("Lifetime generator validation: OK")

if __name__ == "__main__":
    _run_validation_tests()
    print("Lifetime generator module initialized successfully.")