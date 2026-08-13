"""
EIR (Effective Interest Rate) Generator Module
=============================================

V1 EIR Generation Component for Synthetic Loan Dataset

This module implements the V1 methodology for generating synthetic Effective Interest Rates
(EIR) that represent the original contractual economics of loans at origination.

Key Features:
- Generates EIR values based on product taxonomy configuration
- Uses product type as primary determinant of EIR range
- Applies modest stochastic variation around configured base rates
- Ensures EIR values remain within product-specific min/max bounds
- Supports explicit seeding for full reproducibility
- Does NOT use current/future information (PD_current, default status, etc.)
- Does NOT use complicated credit-pricing models

Methodology:
EIR = f(product_type, base_eir, stochastic_variation)
where:
- base_eir comes from product taxonomy configuration
- stochastic_variation is small, bounded, and product-specific
- final EIR is clipped to [min_eir, max_eir] bounds

V1 EIR Configuration (from product_taxonomy.py):
- mortgage: min 4.5%, max 7.5%, base 6.0%
- auto_loan: min 4.0%, max 9.0%, base 6.5%
- credit_card: min 12%, max 25%, base 19%
- student_loan: min 4.0%, max 8.0%, base 5.5%
- other_personal_loan: min 8%, max 22%, base 14%

Important Constraints:
- EIR represents ORIGINAL contractual economics (not current/future state)
- Must NOT use PD_current, current borrower characteristics, default status, staging, LGD, EAD
- Must remain within configured min/max bounds for each product
- Should be continuous (not discrete) and reproducible
- Product type is the PRIMARY determinant of EIR range
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from .product_taxonomy import PRODUCT_TAXONOMY, PRODUCT_TYPES

def generate_eir_for_product(
    product_type: str,
    n_loans: int = 1,
    seed: Optional[int] = None,
    variation_scale: float = 0.5
) -> Union[float, List[float]]:
    """
    Generate EIR values for a specific product type.

    Args:
        product_type: Product type from PRODUCT_TYPES
        n_loans: Number of EIR values to generate
        seed: Random seed for reproducibility
        variation_scale: Scale factor for stochastic variation (0-1 range)

    Returns:
        Single EIR value if n_loans=1, otherwise list of EIR values

    Raises:
        ValueError: If product_type is unknown or parameters are invalid
    """
    # Validate product type
    if product_type not in PRODUCT_TAXONOMY:
        raise ValueError(
            f"Unknown product_type: {product_type!r}. "
            f"Valid types: {sorted(PRODUCT_TYPES)}"
        )

    if n_loans <= 0:
        raise ValueError(f"n_loans must be positive, got {n_loans}")

    if not (0 <= variation_scale <= 1):
        raise ValueError(f"variation_scale must be in [0, 1], got {variation_scale}")

    # Get EIR configuration for the product
    eir_config = PRODUCT_TAXONOMY[product_type]["eir"]
    base_eir = eir_config["base"]
    min_eir = eir_config["min"]
    max_eir = eir_config["max"]

    # Calculate variation range (scaled by product's available range)
    available_range = max_eir - min_eir
    variation_range = available_range * variation_scale

    # Generate stochastic variation using normal distribution
    # centered at 0 with standard deviation proportional to variation_range
    rng = np.random.default_rng(seed)
    if n_loans == 1:
        # Single value case
        stochastic_variation = rng.normal(0, variation_range / 4)
        eir_value = base_eir + stochastic_variation

        # Clip to bounds
        eir_value = np.clip(eir_value, min_eir, max_eir)

        return float(eir_value)
    else:
        # Multiple values case
        stochastic_variations = rng.normal(0, variation_range / 4, size=n_loans)
        eir_values = base_eir + stochastic_variations

        # Clip to bounds
        eir_values = np.clip(eir_values, min_eir, max_eir)

        return [float(val) for val in eir_values]

def generate_eir_for_multiple_products(
    product_assignments: List[str],
    seed: Optional[int] = None,
    variation_scale: float = 0.5
) -> Dict[str, List[float]]:
    """
    Generate EIR values for multiple products based on product assignments.

    Args:
        product_assignments: List of product types (one per loan)
        seed: Random seed for reproducibility
        variation_scale: Scale factor for stochastic variation

    Returns:
        Dictionary mapping product types to lists of EIR values

    Raises:
        ValueError: If product_assignments is empty or contains invalid products
    """
    if not product_assignments:
        raise ValueError("product_assignments cannot be empty")

    if len(product_assignments) != len([p for p in product_assignments if p in PRODUCT_TAXONOMY]):
        invalid_products = set(product_assignments) - PRODUCT_TYPES
        raise ValueError(f"Invalid product types in assignments: {invalid_products}")

    # Group loans by product type
    product_groups = {}
    for product in product_assignments:
        if product not in product_groups:
            product_groups[product] = []
        product_groups[product].append(product)

    # Generate EIR values for each product group
    eir_results = {}
    for product, loans in product_groups.items():
        n_loans = len(loans)
        # Use deterministic seed based on main seed and product for reproducibility
        product_seed = None if seed is None else hash((seed, product)) % 1000000
        eir_values = generate_eir_for_product(
            product_type=product,
            n_loans=n_loans,
            seed=product_seed,
            variation_scale=variation_scale
        )
        # Ensure eir_values is always a list (even for single loans)
        if isinstance(eir_values, float):
            eir_values = [eir_values]
        eir_results[product] = eir_values

    return eir_results

def generate_eir_dataframe(
    product_assignments: List[str],
    seed: Optional[int] = None,
    variation_scale: float = 0.5
) -> pd.DataFrame:
    """
    Generate EIR values and return as DataFrame with product assignments.

    Args:
        product_assignments: List of product types (one per loan)
        seed: Random seed for reproducibility
        variation_scale: Scale factor for stochastic variation

    Returns:
        DataFrame with columns: product_type, eir

    Raises:
        ValueError: If product_assignments is empty or contains invalid products
    """
    if not product_assignments:
        raise ValueError("product_assignments cannot be empty")

    # Generate EIR values
    eir_dict = generate_eir_for_multiple_products(
        product_assignments=product_assignments,
        seed=seed,
        variation_scale=variation_scale
    )

    # Create DataFrame
    all_products = []
    all_eirs = []

    for product, eir_values in eir_dict.items():
        all_products.extend([product] * len(eir_values))
        all_eirs.extend(eir_values)

    # Verify we have the same number of EIR values as product assignments
    if len(all_products) != len(product_assignments):
        raise RuntimeError(
            f"EIR generation error: expected {len(product_assignments)} values, "
            f"got {len(all_products)}"
        )

    return pd.DataFrame({
        "product_type": all_products,
        "eir": all_eirs
    })

def validate_eir_values(
    eir_values: Union[List[float], np.ndarray],
    product_type: str
) -> Dict[str, bool]:
    """
    Validate that EIR values conform to product constraints.

    Args:
        eir_values: List or array of EIR values to validate
        product_type: Product type for validation

    Returns:
        Dictionary of validation results with boolean flags
    """
    # Get EIR configuration for the product
    eir_config = PRODUCT_TAXONOMY[product_type]["eir"]
    min_eir = eir_config["min"]
    max_eir = eir_config["max"]

    validation_results = {}

    # Convert to numpy array for easier validation
    eir_array = np.array(eir_values)

    # Check all values are within bounds
    validation_results["all_within_bounds"] = (
        (eir_array >= min_eir).all() and
        (eir_array <= max_eir).all()
    )

    # Check all values are positive
    validation_results["all_positive"] = (eir_array > 0).all()

    # Check reasonable variation (not all identical unless n=1)
    if len(eir_array) > 1:
        validation_results["has_variation"] = not np.allclose(eir_array, eir_array[0])
    else:
        validation_results["has_variation"] = True  # Single value is fine

    return validation_results

def get_eir_statistics(
    eir_values: Union[List[float], np.ndarray, pd.Series],
    product_type: str
) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for EIR values.

    Args:
        eir_values: EIR values to analyze
        product_type: Product type for context

    Returns:
        Dictionary of statistical measures
    """
    eir_array = np.array(eir_values)
    eir_config = PRODUCT_TAXONOMY[product_type]["eir"]
    base_eir = eir_config["base"]

    return {
        "count": len(eir_array),
        "min": float(np.min(eir_array)),
        "median": float(np.median(eir_array)),
        "mean": float(np.mean(eir_array)),
        "max": float(np.max(eir_array)),
        "std": float(np.std(eir_array)),
        "base_eir": float(base_eir),
        "distance_from_base": float(np.mean(np.abs(eir_array - base_eir))),
        "coefficient_of_variation": float(np.std(eir_array) / np.mean(eir_array)) if np.mean(eir_array) > 0 else 0.0
    }

def validate_eir_reproducibility(
    product_assignments: List[str],
    seed: int,
    n_trials: int = 3,
    variation_scale: float = 0.5
) -> bool:
    """
    Validate that EIR generation is reproducible with the same seed.

    Args:
        product_assignments: List of product types
        seed: Seed to use for reproducibility test
        n_trials: Number of trials to run
        variation_scale: Scale factor for stochastic variation

    Returns:
        True if all trials produce identical results, False otherwise
    """
    # Generate multiple EIR dataframes with the same seed
    eir_dfs = []
    for i in range(n_trials):
        eir_df = generate_eir_dataframe(
            product_assignments=product_assignments,
            seed=seed,
            variation_scale=variation_scale
        )
        eir_dfs.append(eir_df)

    # Check that all dataframes are identical
    reference = eir_dfs[0]
    for i, eir_df in enumerate(eir_dfs[1:], 1):
        if not eir_df.equals(reference):
            print(f"Reproducibility check failed: trial {i} differs from reference")
            return False

    print(f"EIR reproducibility validated: {n_trials} trials with seed {seed} produced identical results")
    return True

def generate_eir_report(
    product_assignments: List[str],
    seed: Optional[int] = None,
    variation_scale: float = 0.5
) -> Dict[str, Any]:
    """
    Generate comprehensive EIR report with statistics and validation.

    Args:
        product_assignments: List of product types
        seed: Random seed for reproducibility
        variation_scale: Scale factor for stochastic variation

    Returns:
        Dictionary containing EIR data, statistics, and validation results
    """
    # Generate EIR values
    eir_df = generate_eir_dataframe(
        product_assignments=product_assignments,
        seed=seed,
        variation_scale=variation_scale
    )

    # Calculate statistics by product
    statistics = {}
    validation = {}

    for product in eir_df["product_type"].unique():
        product_data = eir_df[eir_df["product_type"] == product]
        product_eirs = product_data["eir"].values

        # Calculate statistics
        statistics[product] = get_eir_statistics(product_eirs, product)

        # Validate values
        validation[product] = validate_eir_values(product_eirs, product)

    # Overall validation
    overall_validation = {
        "all_products_valid": all(
            stats["all_within_bounds"] and stats["all_positive"]
            for stats in validation.values()
        ),
        "product_count": len(statistics),
        "total_loans": len(eir_df)
    }

    return {
        "eir_data": eir_df,
        "statistics": statistics,
        "validation": validation,
        "overall_validation": overall_validation,
        "metadata": {
            "seed": seed,
            "variation_scale": variation_scale,
            "generation_timestamp": pd.Timestamp.now(),
            "product_distribution": eir_df["product_type"].value_counts(normalize=True).to_dict()
        }
    }

# Internal validation tests
def _run_validation_tests() -> None:
    """Run internal validation tests for the EIR generator."""
    print("Running EIR generator validation tests...")

    # Test 1: Single product EIR generation
    mortgage_eir = generate_eir_for_product("mortgage", n_loans=1, seed=42)
    assert isinstance(mortgage_eir, float), "Single EIR should be float"
    assert 0.045 <= mortgage_eir <= 0.075, "Mortgage EIR should be within bounds"

    # Test 2: Multiple EIR values for same product
    auto_eirs = generate_eir_for_product("auto_loan", n_loans=10, seed=42)
    assert isinstance(auto_eirs, list), "Multiple EIRs should be list"
    assert len(auto_eirs) == 10, "Should generate correct number of EIRs"
    assert all(0.04 <= eir <= 0.09 for eir in auto_eirs), "Auto EIRs should be within bounds"

    # Test 3: Multiple products
    product_assignments = ["credit_card", "mortgage", "auto_loan", "credit_card"]
    eir_dict = generate_eir_for_multiple_products(product_assignments, seed=42)
    assert set(eir_dict.keys()) == {"credit_card", "mortgage", "auto_loan"}, "Should have all products"
    assert len(eir_dict["credit_card"]) == 2, "Credit card should have 2 EIRs"
    assert len(eir_dict["mortgage"]) == 1, "Mortgage should have 1 EIR"
    assert len(eir_dict["auto_loan"]) == 1, "Auto loan should have 1 EIR"

    # Test 4: DataFrame generation
    eir_df = generate_eir_dataframe(product_assignments, seed=42)
    assert len(eir_df) == len(product_assignments), "DataFrame should have same length as assignments"
    assert "product_type" in eir_df.columns, "Should have product_type column"
    assert "eir" in eir_df.columns, "Should have eir column"

    # Test 5: Validation
    validation = validate_eir_values(auto_eirs, "auto_loan")
    assert validation["all_within_bounds"], "All EIRs should be within bounds"
    assert validation["all_positive"], "All EIRs should be positive"

    # Test 6: Statistics
    stats = get_eir_statistics(auto_eirs, "auto_loan")
    assert "mean" in stats, "Statistics should include mean"
    assert "std" in stats, "Statistics should include std"
    assert "base_eir" in stats, "Statistics should include base_eir"

    # Test 7: Reproducibility
    assert validate_eir_reproducibility(["mortgage", "auto_loan"], seed=123, n_trials=2), "Should be reproducible"

    # Test 8: Error handling
    try:
        generate_eir_for_product("invalid_product", n_loans=1)
        assert False, "Should raise error for invalid product"
    except ValueError:
        pass  # Expected

    try:
        generate_eir_for_product("mortgage", n_loans=0)
        assert False, "Should raise error for n_loans=0"
    except ValueError:
        pass  # Expected

    # Test 9: Report generation
    report = generate_eir_report(product_assignments, seed=42)
    assert "eir_data" in report, "Report should contain eir_data"
    assert "statistics" in report, "Report should contain statistics"
    assert "validation" in report, "Report should contain validation"

    print("EIR generator validation: OK")

if __name__ == "__main__":
    _run_validation_tests()
    print("EIR generator module initialized successfully.")