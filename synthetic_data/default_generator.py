"""
Default Generator Module for Synthetic Loan Dataset
==================================================

V1 Default Generation Component for Synthetic Loan Dataset

This module implements the V1 methodology for generating synthetic default status
values using Bernoulli realization based on PD_current probabilities.

Key Features:
- Generates default_status values using Bernoulli trials: default_status ~ Bernoulli(PD_current)
- Uses only PD_current and explicit random seed as inputs
- Does NOT use PD_origin, original GMSC SeriousDlqin2yrs, or any other variables
- Does NOT modify PD values or introduce thresholds
- Preserves exact PD_current values as probabilities
- Supports single/batch generation with explicit seeds
- Provides comprehensive validation and statistics
- Fully reproducible with explicit seeds
- Independent from staging and ECL components

Methodology:
default_status_i ~ Bernoulli(PD_current_i)

where:
- PD_current represents the predicted probability of default
- default_status represents the synthetic observed outcome (0 or 1)
- Each loan receives exactly one independent Bernoulli realization
- P(default_status = 1) = PD_current
- P(default_status = 0) = 1 - PD_current

Important Distinction:
- PD = predicted probability of default (continuous [0,1])
- default_status = synthetic observed outcome (binary {0,1})
- The Bernoulli draw is the bridge between the two

Critical Validation:
For a generated portfolio:
- expected_defaults = sum(PD_current)
- actual_defaults = sum(default_status)
- actual_defaults should naturally fluctuate around expected_defaults
- This is a random Bernoulli realization, not exact matching
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

def _validate_pd_values(pd_values: Union[List[float], np.ndarray]) -> None:
    """
    Validate that PD values are within valid probability range [0, 1].

    Args:
        pd_values: Array of PD values to validate

    Raises:
        ValueError: If any PD value is outside [0, 1] or if array is empty
    """
    pd_array = np.array(pd_values)

    if len(pd_array) == 0:
        raise ValueError("PD values array cannot be empty")

    if not np.all((pd_array >= 0) & (pd_array <= 1)):
        invalid_mask = (pd_array < 0) | (pd_array > 1)
        invalid_values = pd_array[invalid_mask]
        raise ValueError(
            f"PD values must be in [0, 1] range. "
            f"Found invalid values: {invalid_values[:10]}... "
            f"(showing first 10 of {len(invalid_values)})"
        )

def generate_default_status_single(
    pd_current: float,
    seed: Optional[int] = None
) -> int:
    """
    Generate default status for a single loan using Bernoulli trial.

    Args:
        pd_current: Predicted probability of default (must be in [0, 1])
        seed: Random seed for reproducibility

    Returns:
        Default status: 1 (default) or 0 (non-default)

    Raises:
        ValueError: If pd_current is not in [0, 1]
    """
    # Validate PD value
    if not (0 <= pd_current <= 1):
        raise ValueError(f"PD_current must be in [0, 1], got {pd_current}")

    # Generate Bernoulli trial
    rng = np.random.default_rng(seed)
    default_status = rng.binomial(1, pd_current)

    return int(default_status)

def generate_default_status_batch(
    pd_current_values: Union[List[float], np.ndarray],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate default status for multiple loans using Bernoulli trials.

    Args:
        pd_current_values: Array of predicted probabilities of default
        seed: Random seed for reproducibility

    Returns:
        Array of default statuses: 1 (default) or 0 (non-default)

    Raises:
        ValueError: If any pd_current value is outside [0, 1] or array is empty
    """
    # Validate PD values
    _validate_pd_values(pd_current_values)

    # Convert to numpy array for processing
    pd_array = np.array(pd_current_values)

    # Generate Bernoulli trials
    rng = np.random.default_rng(seed)
    default_statuses = rng.binomial(1, pd_array)

    return default_statuses.astype(int)

def generate_default_dataframe(
    pd_current_values: Union[List[float], np.ndarray],
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate default statuses and return as DataFrame.

    Args:
        pd_current_values: Array of predicted probabilities of default
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: pd_current, default_status

    Raises:
        ValueError: If any pd_current value is outside [0, 1] or array is empty
    """
    # Validate PD values
    _validate_pd_values(pd_current_values)

    # Convert to numpy array
    pd_array = np.array(pd_current_values)

    # Generate default statuses
    default_statuses = generate_default_status_batch(pd_array, seed)

    return pd.DataFrame({
        "pd_current": pd_array,
        "default_status": default_statuses
    })

def validate_default_generation(
    pd_current_values: Union[List[float], np.ndarray],
    default_statuses: Union[List[int], np.ndarray]
) -> Dict[str, Any]:
    """
    Validate that default statuses conform to requirements.

    Args:
        pd_current_values: Array of PD values used for generation
        default_statuses: Array of generated default statuses

    Returns:
        Dictionary of validation results and statistics

    Raises:
        ValueError: If validation fails
    """
    # Convert to numpy arrays
    pd_array = np.array(pd_current_values)
    default_array = np.array(default_statuses)

    # Check lengths match
    if len(pd_array) != len(default_array):
        raise ValueError(
            f"Length mismatch: PD values ({len(pd_array)}) "
            f"!= default statuses ({len(default_array)})"
        )

    # Check default statuses are binary
    if not np.all((default_array == 0) | (default_array == 1)):
        invalid_values = np.unique(default_array[~((default_array == 0) | (default_array == 1))])
        raise ValueError(
            f"Default statuses must be binary {0, 1}. "
            f"Found invalid values: {invalid_values}"
        )

    # Calculate statistics
    expected_defaults = np.sum(pd_array)
    actual_defaults = np.sum(default_array)
    default_rate = actual_defaults / len(default_array)
    expected_default_rate = expected_defaults / len(default_array)

    return {
        "validation": {
            "all_binary": True,
            "length_match": True,
            "pd_range_valid": True
        },
        "statistics": {
            "count": len(default_array),
            "expected_defaults": float(expected_defaults),
            "actual_defaults": float(actual_defaults),
            "expected_default_rate": float(expected_default_rate),
            "actual_default_rate": float(default_rate),
            "difference": float(actual_defaults - expected_defaults),
            "relative_difference": float((actual_defaults - expected_defaults) / max(expected_defaults, 1))
        },
        "pd_statistics": {
            "pd_min": float(pd_array.min()),
            "pd_max": float(pd_array.max()),
            "pd_mean": float(pd_array.mean()),
            "pd_median": float(np.median(pd_array)),
            "pd_std": float(pd_array.std())
        }
    }

def get_default_statistics(
    default_results: Dict[str, Any]
) -> Dict[str, float]:
    """
    Extract and format key statistics from validation results.

    Args:
        default_results: Dictionary from validate_default_generation

    Returns:
        Dictionary of key statistical measures
    """
    stats = default_results["statistics"]
    pd_stats = default_results["pd_statistics"]

    return {
        "count": stats["count"],
        "expected_defaults": stats["expected_defaults"],
        "actual_defaults": stats["actual_defaults"],
        "expected_default_rate": stats["expected_default_rate"],
        "actual_default_rate": stats["actual_default_rate"],
        "difference": stats["difference"],
        "relative_difference": stats["relative_difference"],
        "pd_min": pd_stats["pd_min"],
        "pd_max": pd_stats["pd_max"],
        "pd_mean": pd_stats["pd_mean"],
        "pd_median": pd_stats["pd_median"],
        "pd_std": pd_stats["pd_std"]
    }

def validate_default_reproducibility(
    pd_current_values: Union[List[float], np.ndarray],
    seed: int,
    n_trials: int = 3
) -> bool:
    """
    Validate that default generation is reproducible with the same seed.

    Args:
        pd_current_values: Array of PD values
        seed: Seed to use for reproducibility test
        n_trials: Number of trials to run

    Returns:
        True if all trials produce identical results, False otherwise
    """
    # Generate multiple default arrays with the same seed
    default_arrays = []
    for i in range(n_trials):
        defaults = generate_default_status_batch(pd_current_values, seed=seed)
        default_arrays.append(defaults)

    # Check that all arrays are identical
    reference = default_arrays[0]
    for i, defaults in enumerate(default_arrays[1:], 1):
        if not np.array_equal(defaults, reference):
            print(f"Reproducibility check failed: trial {i} differs from reference")
            return False

    print(f"Default reproducibility validated: {n_trials} trials with seed {seed} produced identical results")
    return True

def test_bernoulli_convergence(
    pd_value: float,
    n_samples: int = 10000,
    n_trials: int = 100,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Test Bernoulli convergence by generating many realizations for a fixed PD.

    Args:
        pd_value: Fixed PD value to test (e.g., 0.10)
        n_samples: Number of samples per trial
        n_trials: Number of trials to run
        seed: Base seed for reproducibility

    Returns:
        Dictionary containing convergence statistics
    """
    if not (0 <= pd_value <= 1):
        raise ValueError(f"PD value must be in [0, 1], got {pd_value}")

    # Generate multiple trials
    empirical_rates = []
    rng = np.random.default_rng(seed)

    for trial in range(n_trials):
        # Use different seed for each trial
        trial_seed = None if seed is None else seed + trial
        defaults = generate_default_status_batch([pd_value] * n_samples, seed=trial_seed)
        empirical_rate = np.sum(defaults) / n_samples
        empirical_rates.append(empirical_rate)

    # Calculate statistics
    mean_rate = np.mean(empirical_rates)
    std_rate = np.std(empirical_rates)
    min_rate = np.min(empirical_rates)
    max_rate = np.max(empirical_rates)
    error_from_target = abs(mean_rate - pd_value)

    return {
        "target_pd": float(pd_value),
        "mean_empirical_rate": float(mean_rate),
        "std_empirical_rate": float(std_rate),
        "min_empirical_rate": float(min_rate),
        "max_empirical_rate": float(max_rate),
        "error_from_target": float(error_from_target),
        "convergence_quality": "GOOD" if error_from_target < pd_value * 0.1 else "FAIR",
        "n_trials": n_trials,
        "n_samples_per_trial": n_samples
    }

def generate_default_report(
    pd_current_values: Union[List[float], np.ndarray],
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive default report with statistics and validation.

    Args:
        pd_current_values: Array of PD values
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing default data, statistics, and validation results
    """
    # Generate default statuses
    default_df = generate_default_dataframe(pd_current_values, seed=seed)

    # Validate and get statistics
    validation_results = validate_default_generation(
        default_df["pd_current"].values,
        default_df["default_status"].values
    )

    # Test convergence for a few PD values
    convergence_tests = {
        "low_pd": test_bernoulli_convergence(0.05, n_samples=10000, n_trials=50, seed=seed),
        "medium_pd": test_bernoulli_convergence(0.10, n_samples=10000, n_trials=50, seed=seed),
        "high_pd": test_bernoulli_convergence(0.20, n_samples=10000, n_trials=50, seed=seed)
    }

    return {
        "default_data": default_df,
        "validation": validation_results,
        "convergence_tests": convergence_tests,
        "metadata": {
            "seed": seed,
            "generation_timestamp": pd.Timestamp.now(),
            "pd_value_count": len(pd_current_values),
            "pd_value_range": f"[{default_df['pd_current'].min():.4f}, {default_df['pd_current'].max():.4f}]"
        }
    }

def _run_validation_tests() -> None:
    """Run internal validation tests for the default generator."""
    print("Running default generator validation tests...")

    # Test 1: Single loan generation
    for pd_val in [0.01, 0.05, 0.10, 0.20, 0.50, 0.99]:
        status = generate_default_status_single(pd_val, seed=42)
        assert status in [0, 1], f"Single default status should be 0 or 1, got {status}"

    # Test 2: Batch generation
    pd_values = [0.01, 0.05, 0.10, 0.20, 0.50]
    defaults = generate_default_status_batch(pd_values, seed=42)
    assert len(defaults) == len(pd_values), "Batch generation should preserve length"
    assert np.all((defaults == 0) | (defaults == 1)), "All default statuses should be binary"

    # Test 3: DataFrame generation
    df = generate_default_dataframe(pd_values, seed=42)
    assert "pd_current" in df.columns, "DataFrame should have pd_current column"
    assert "default_status" in df.columns, "DataFrame should have default_status column"
    assert len(df) == len(pd_values), "DataFrame should have correct length"

    # Test 4: Validation
    validation = validate_default_generation(pd_values, defaults)
    assert validation["validation"]["all_binary"], "Validation should confirm binary statuses"
    assert validation["statistics"]["count"] == len(pd_values), "Count should match input size"

    # Test 5: Reproducibility
    assert validate_default_reproducibility(pd_values, seed=123, n_trials=2), "Should be reproducible"

    # Test 6: Error handling - invalid PD values
    try:
        generate_default_status_single(1.5, seed=42)
        assert False, "Should raise error for PD > 1"
    except ValueError:
        pass  # Expected

    try:
        generate_default_status_single(-0.1, seed=42)
        assert False, "Should raise error for PD < 0"
    except ValueError:
        pass  # Expected

    try:
        generate_default_status_batch([], seed=42)
        assert False, "Should raise error for empty array"
    except ValueError:
        pass  # Expected

    # Test 7: Bernoulli convergence
    convergence = test_bernoulli_convergence(0.10, n_samples=1000, n_trials=30, seed=42)
    assert convergence["error_from_target"] < 0.02, "Convergence should be reasonable"

    # Test 8: Report generation
    report = generate_default_report(pd_values, seed=42)
    assert "default_data" in report, "Report should contain default_data"
    assert "validation" in report, "Report should contain validation"
    assert "convergence_tests" in report, "Report should contain convergence_tests"

    print("Default generator validation: OK")

if __name__ == "__main__":
    _run_validation_tests()
    print("Default generator module initialized successfully.")