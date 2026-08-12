import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import warnings

# Default configuration for trajectory-based modifications
DEFAULT_CONFIG = {
    # Trajectory probabilities
    'trajectory_probs': {
        'stable': 0.60,
        'improving': 0.25,
        'deteriorating': 0.15
    },

    # Variable-specific modification parameters
    'variables': {
        'MonthlyIncome': {
            'improvement_direction': 'positive',
            'stable_mean': 0.05, 'stable_std': 0.04,
            'improving_mean': 0.15, 'improving_std': 0.06,
            'deteriorating_mean': -0.03, 'deteriorating_std': 0.04,
            'min_value': 0.0
        },
        'RevolvingUtilizationOfUnsecuredLines': {
            'improvement_direction': 'negative',
            'stable_mean': 0.0, 'stable_std': 0.05,
            'improving_mean': -0.15, 'improving_std': 0.10,
            'deteriorating_mean': 0.10, 'deteriorating_std': 0.08,
            'min_value': 0.0, 'max_value': 1.0
        },
        'DebtRatio': {
            'improvement_direction': 'negative',
            'stable_mean': 0.0, 'stable_std': 0.05,
            'improving_mean': -0.10, 'improving_std': 0.08,
            'deteriorating_mean': 0.08, 'deteriorating_std': 0.06,
            'min_value': 0.0
        },
        'NumberOfTime30-59DaysPastDueNotWorse': {
            'improvement_direction': 'negative',
            'stable_mean': 0.0, 'stable_std': 0.5,
            'improving_mean': -0.3, 'improving_std': 0.4,
            'deteriorating_mean': 0.4, 'deteriorating_std': 0.5,
            'min_value': 0.0
        },
        'NumberOfTimes90DaysLate': {
            'improvement_direction': 'negative',
            'stable_mean': 0.0, 'stable_std': 0.3,
            'improving_mean': -0.2, 'improving_std': 0.3,
            'deteriorating_mean': 0.3, 'deteriorating_std': 0.4,
            'min_value': 0.0
        },
        'NumberOfTime60-89DaysPastDueNotWorse': {
            'improvement_direction': 'negative',
            'stable_mean': 0.0, 'stable_std': 0.4,
            'improving_mean': -0.2, 'improving_std': 0.3,
            'deteriorating_mean': 0.3, 'deteriorating_std': 0.4,
            'min_value': 0.0
        },
        'NumberOfOpenCreditLinesAndLoans': {
            'improvement_direction': 'neutral',
            'stable_mean': 0.0, 'stable_std': 0.3,
            'improving_mean': 0.1, 'improving_std': 0.2,
            'deteriorating_mean': -0.1, 'deteriorating_std': 0.2,
            'min_value': 0.0
        },
        'NumberRealEstateLoansOrLines': {
            'improvement_direction': 'neutral',
            'stable_mean': 0.0, 'stable_std': 0.2,
            'improving_mean': 0.05, 'improving_std': 0.15,
            'deteriorating_mean': -0.05, 'deteriorating_std': 0.15,
            'min_value': 0.0
        },
        'NumberOfDependents': {
            'improvement_direction': 'neutral',
            'stable_mean': 0.0, 'stable_std': 0.1,
            'improving_mean': 0.0, 'improving_std': 0.05,
            'deteriorating_mean': 0.0, 'deteriorating_std': 0.05,
            'min_value': 0.0
        }
    },

    # Count variables (require integer values)
    'count_variables': [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberRealEstateLoansOrLines'
    ]
}

def _validate_input_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input dataframe contains required columns."""
    required_columns = [
        'SeriousDlqin2yrs',
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def _merge_config(base_config: dict, override_config: Optional[dict] = None) -> dict:
    """Merge configuration with overrides."""
    if override_config is None:
        return base_config

    merged = base_config.copy()
    if 'trajectory_probs' in override_config:
        merged['trajectory_probs'] = override_config['trajectory_probs']

    if 'variables' in override_config:
        merged['variables'] = base_config['variables'].copy()
        merged['variables'].update(override_config['variables'])

    return merged

def _assign_trajectories(rng: np.random.Generator, n_rows: int, trajectory_probs: dict) -> np.array:
    """Assign trajectories to each borrower using multinomial distribution."""
    trajectories = list(trajectory_probs.keys())
    probs = list(trajectory_probs.values())

    # Normalize probabilities to ensure they sum to 1
    probs = np.array(probs) / np.sum(probs)

    return rng.choice(trajectories, size=n_rows, p=probs)

def _apply_continuous_modification(
    series: pd.Series,
    trajectories: np.array,
    config: dict,
    rng: np.random.Generator
) -> pd.Series:
    """
    Apply trajectory-specific modification to continuous variables.

    Args:
        series: Original values
        trajectories: Array of trajectory assignments for each row
        config: Variable-specific configuration
        rng: Random number generator

    Returns:
        Modified series with applied shocks
    """
    modified = series.copy().astype(float)

    for i, trajectory in enumerate(trajectories):
        # Get trajectory-specific parameters
        mean = config[f'{trajectory}_mean']
        std = config[f'{trajectory}_std']

        # Generate shock for this specific row
        shock = rng.normal(mean, std)

        # Apply shock based on improvement direction
        if config['improvement_direction'] == 'positive':
            # Higher values = improvement (e.g., income)
            modified.iloc[i] = series.iloc[i] * (1 + shock)
        elif config['improvement_direction'] == 'negative':
            # Lower values = improvement (e.g., utilization, debt ratio)
            # For negative direction variables, we add the shock (which can be negative for improving)
            modified.iloc[i] = series.iloc[i] * (1 + shock)
        else:  # neutral
            # Small changes in either direction
            modified.iloc[i] = series.iloc[i] * (1 + shock * 0.5)

    # Apply bounds
    if 'min_value' in config:
        modified = np.maximum(modified, config['min_value'])
    if 'max_value' in config:
        modified = np.minimum(modified, config['max_value'])

    return modified

def _apply_count_modification(
    series: pd.Series,
    trajectories: np.array,
    config: dict,
    rng: np.random.Generator
) -> pd.Series:
    """
    Apply trajectory-specific modification to count variables.

    Args:
        series: Original count values
        trajectories: Array of trajectory assignments for each row
        config: Variable-specific configuration
        rng: Random number generator

    Returns:
        Modified series with integer count values
    """
    modified = series.copy().astype(float)

    for i, trajectory in enumerate(trajectories):
        # Get trajectory-specific parameters
        mean = config[f'{trajectory}_mean']
        std = config[f'{trajectory}_std']

        # Generate shock for this specific row
        shock = rng.normal(mean, std)

        # Apply shock based on improvement direction
        if config['improvement_direction'] == 'negative':
            # Lower counts = improvement (e.g., delinquencies)
            modified.iloc[i] = series.iloc[i] + shock
        elif config['improvement_direction'] == 'positive':
            # Higher counts = improvement
            modified.iloc[i] = series.iloc[i] + shock
        else:  # neutral
            # Small changes in either direction
            modified.iloc[i] = series.iloc[i] + shock * 0.5

    # Round to nearest integer and apply bounds
    # Handle NaN values by filling with 0 (since these are count variables)
    modified = np.round(modified).fillna(0).astype(int)

    if 'min_value' in config:
        modified = np.maximum(modified, config['min_value'])
    if 'max_value' in config:
        modified = np.minimum(modified, config['max_value'])

    return modified

def construct_current_snapshot(
    origination_df: pd.DataFrame,
    seed: Optional[int] = None,
    trajectory_probs: Optional[dict] = None,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Construct current borrower snapshot from origination snapshot.

    This function creates a modified version of the origination snapshot to simulate
    borrower characteristics approximately 2 years later, based on assigned trajectories.

    Args:
        origination_df: DataFrame containing origination borrower data with required columns
        seed: Random seed for reproducibility (default: None)
        trajectory_probs: Optional override for trajectory probabilities
        config: Optional configuration override

    Returns:
        DataFrame with modified current snapshot, preserving original index and adding trajectory column

    Raises:
        ValueError: If required columns are missing from input dataframe
    """
    # Validate input
    _validate_input_dataframe(origination_df)

    # Merge configuration
    merged_config = _merge_config(DEFAULT_CONFIG, config)

    # Override trajectory probabilities if provided
    if trajectory_probs is not None:
        merged_config['trajectory_probs'] = trajectory_probs

    # Create copy to avoid mutating original
    current_df = origination_df.copy()

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Assign trajectories
    n_rows = len(current_df)
    trajectories = _assign_trajectories(rng, n_rows, merged_config['trajectory_probs'])
    current_df['trajectory'] = trajectories

    # Advance age by exactly 2 years (deterministic)
    current_df['age'] = current_df['age'] + 2

    # Apply modifications to each variable
    for var_name, var_config in merged_config['variables'].items():
        if var_name not in current_df.columns:
            warnings.warn(f"Configuration specifies variable {var_name} but it's not in the dataframe")
            continue

        if var_name in merged_config['count_variables']:
            # Handle count variables
            current_df[var_name] = _apply_count_modification(
                current_df[var_name],
                trajectories,
                var_config,
                rng
            )
        else:
            # Handle continuous variables
            current_df[var_name] = _apply_continuous_modification(
                current_df[var_name],
                trajectories,
                var_config,
                rng
            )

    # Preserve SeriousDlqin2yrs for lineage but don't modify it
    # (it's the target variable, not a borrower characteristic)

    return current_df