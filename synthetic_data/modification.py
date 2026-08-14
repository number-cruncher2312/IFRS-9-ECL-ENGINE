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
    ],

    # Zero-income transition configuration
    # These are V1 synthetic assumptions, not empirical probabilities
    'zero_income_transition': {
        'improving_probability': 0.7,  # High chance of getting income when improving
        'stable_probability': 0.2,     # Low chance of getting income when stable
        'deteriorating_probability': 0.1, # Very low chance when deteriorating
        'positive_income_distribution': {
            'mean': 3000,
            'std': 1500,
            'min_value': 1000,
            'max_value': 8000
        }
    },

    # Special DebtRatio handling for zero-income borrowers
    'debt_ratio_zero_income': {
        'improving_mean': -0.2, 'improving_std': 0.1,
        'stable_mean': 0.0, 'stable_std': 0.05,
        'deteriorating_mean': 0.1, 'deteriorating_std': 0.1,
        'min_value': 0.0,
        'max_value': 4.0  # Conservative upper bound to prevent unreasonable values
    }
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
        # Skip modification for stable entries
        if trajectory == 'stable':
            continue
            
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

def _generate_positive_income(
    n_samples: int,
    config: dict,
    rng: np.random.Generator
) -> np.array:
    """
    Generate plausible positive income values for zero-income borrowers who transition to positive income.

    Args:
        n_samples: Number of positive income values to generate
        config: Zero-income transition configuration
        rng: Random number generator

    Returns:
        Array of generated positive income values
    """
    if n_samples == 0:
        return np.array([])

    pos_config = config['positive_income_distribution']

    # Generate income values from normal distribution
    incomes = rng.normal(pos_config['mean'], pos_config['std'], n_samples)

    # Apply bounds
    if 'min_value' in pos_config:
        incomes = np.maximum(incomes, pos_config['min_value'])
    if 'max_value' in pos_config:
        incomes = np.minimum(incomes, pos_config['max_value'])

    return incomes

def _apply_zero_income_transition(
    series: pd.Series,
    trajectories: np.array,
    zero_income_config: dict,
    rng: np.random.Generator
) -> pd.Series:
    """
    Apply zero-income specific transition logic.

    For borrowers with MonthlyIncome == 0 at origination, determine whether they:
    - Remain at zero income, or
    - Transition to positive income

    Args:
        series: Original MonthlyIncome values
        trajectories: Array of trajectory assignments for each row
        zero_income_config: Zero-income transition configuration
        rng: Random number generator

    Returns:
        Modified series with zero-income transitions applied
    """
    modified = series.copy().astype(float)

    # Identify zero-income rows
    zero_income_mask = series == 0
    zero_income_indices = np.where(zero_income_mask)[0]

    if len(zero_income_indices) == 0:
        return modified

    # For each zero-income borrower, determine if they transition to positive income
    transition_probs = []
    for i in zero_income_indices:
        trajectory = trajectories[i]
        prob_key = f'{trajectory}_probability'
        transition_probs.append(zero_income_config[prob_key])

    # Determine which zero-income borrowers transition to positive income
    transition_decisions = rng.random(len(zero_income_indices)) < transition_probs

    # Generate positive income for those who transition
    n_transitions = np.sum(transition_decisions)
    if n_transitions > 0:
        positive_incomes = _generate_positive_income(n_transitions, zero_income_config, rng)
        transition_idx = 0
        for i, will_transition in zip(zero_income_indices, transition_decisions):
            if will_transition:
                modified.iloc[i] = positive_incomes[transition_idx]
                transition_idx += 1

    return modified

def _apply_debt_ratio_zero_income(
    series: pd.Series,
    trajectories: np.array,
    config: dict,
    rng: np.random.Generator
) -> pd.Series:
    """
    Apply special DebtRatio modification for zero-income borrowers.

    Args:
        series: Original DebtRatio values
        trajectories: Array of trajectory assignments for each row
        config: DebtRatio configuration for zero-income borrowers
        rng: Random number generator

    Returns:
        Modified series with applied shocks for zero-income borrowers
    """
    modified = series.copy().astype(float)

    # Identify zero-income rows (we'll need to know this from MonthlyIncome)
    # This function will be called after MonthlyIncome is processed, so we need
    # to identify zero-income borrowers differently. For now, we'll use a simple approach.

    for i, trajectory in enumerate(trajectories):
        # Get trajectory-specific parameters
        mean = config[f'{trajectory}_mean']
        std = config[f'{trajectory}_std']

        # Generate shock for this specific row
        shock = rng.normal(mean, std)

        # Apply additive shock (not multiplicative) to avoid explosions
        modified.iloc[i] = series.iloc[i] + shock

    # Apply bounds
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
            # Handle continuous variables with special logic for zero-income cases
            if var_name == 'MonthlyIncome':
                # Apply zero-income specific transition logic first
                current_df[var_name] = _apply_zero_income_transition(
                    current_df[var_name],
                    trajectories,
                    merged_config['zero_income_transition'],
                    rng
                )

                # Then apply regular continuous modification to non-zero income borrowers
                # We need to separate zero and non-zero cases
                non_zero_mask = current_df[var_name] > 0
                if non_zero_mask.any():
                    non_zero_modified = _apply_continuous_modification(
                        current_df[var_name][non_zero_mask],
                        trajectories[non_zero_mask],
                        var_config,
                        rng
                    )
                    current_df.loc[non_zero_mask, var_name] = non_zero_modified

            elif var_name == 'DebtRatio':
                # Apply special DebtRatio handling for zero-income borrowers
                # First identify zero-income borrowers from the original data
                zero_income_mask = origination_df['MonthlyIncome'] == 0

                if zero_income_mask.any():
                    # Apply special handling for zero-income borrowers
                    debt_ratio_zero_income = _apply_debt_ratio_zero_income(
                        current_df[var_name][zero_income_mask],
                        trajectories[zero_income_mask],
                        merged_config['debt_ratio_zero_income'],
                        rng
                    )
                    current_df.loc[zero_income_mask, var_name] = debt_ratio_zero_income

                if (~zero_income_mask).any():
                    # Apply regular continuous modification for non-zero income borrowers
                    debt_ratio_regular = _apply_continuous_modification(
                        current_df[var_name][~zero_income_mask],
                        trajectories[~zero_income_mask],
                        var_config,
                        rng
                    )
                    current_df.loc[~zero_income_mask, var_name] = debt_ratio_regular
            else:
                # Handle other continuous variables normally
                current_df[var_name] = _apply_continuous_modification(
                    current_df[var_name],
                    trajectories,
                    var_config,
                    rng
                )

    # Preserve SeriousDlqin2yrs for lineage but don't modify it
    # (it's the target variable, not a borrower characteristic)

    return current_df