#!/usr/bin/env python3
"""
Standalone script to process GMSC CSV files and generate modified current snapshots.

Usage:
    python process_gmsc_snapshot.py input.csv output.csv [seed]

Example:
    python process_gmsc_snapshot.py input.csv output.csv 42
"""

import pandas as pd
import sys
import os

# Add the project root to Python path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.modification import construct_current_snapshot

def main():
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python process_gmsc_snapshot.py input.csv output.csv [seed]")
        print("Example: python process_gmsc_snapshot.py input.csv output.csv 42")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None

    print(f"Processing GMSC snapshot...")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Seed:   {seed if seed is not None else 'None (random)'}")
    print()

    try:
        # Load input data
        print("Loading input data...")
        origination_df = pd.read_csv(input_path)
        print(f"   Loaded {len(origination_df)} borrowers with {len(origination_df.columns)} columns")

        # Generate current snapshot
        print("Generating current snapshot...")
        current_df = construct_current_snapshot(origination_df, seed=seed)

        # Save results
        print("Saving output...")
        current_df.to_csv(output_path, index=False)
        print(f"   Saved {len(current_df)} borrowers to {output_path}")

        # Show summary statistics
        print("Summary:")
        trajectory_counts = current_df['trajectory'].value_counts()
        for trajectory, count in trajectory_counts.items():
            print(f"   {trajectory:15s}: {count:5d} ({count/len(current_df)*100:5.1f}%)")

        # Verify age advancement
        age_increase = (current_df['age'] - origination_df['age']).eq(2).mean() * 100
        print(f"   Age +2 years:     {age_increase:5.1f}%")

        print("Processing completed successfully!")

    except FileNotFoundError:
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()