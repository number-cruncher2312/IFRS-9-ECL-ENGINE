import pandas as pd
import os
import numpy as np

def extract_csv_snapshot():
    # Define paths
    source_path = r'IFRS9-ECL-Engine\data\GiveMeSomeCredit-training.csv'
    output_folder = r'synthetic_data\credit_snapshot'
    output_path = os.path.join(output_folder, 'GiveMeSomeCredit-snapshot.csv')

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the source CSV file
    print("Reading source CSV file...")
    df = pd.read_csv(source_path)

    # Drop any unnamed columns that might have been added
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Calculate how many rows to extract (10-15k, excluding header)
    total_rows = len(df)
    snapshot_size = min(15000, total_rows)  # Take up to 15k rows

    print(f"Source file has {total_rows} rows (excluding header)")
    print(f"Extracting {snapshot_size} rows for snapshot...")

    # Take a random sample to ensure diversity in the snapshot
    df_snapshot = df.sample(n=snapshot_size, random_state=42)

    # Analyze income data in the snapshot DataFrame
    print("\nAnalyzing income data in snapshot...")
    zero_count = (df_snapshot['MonthlyIncome'] == 0).sum()
    none_count = df_snapshot['MonthlyIncome'].isna().sum()
    total_snapshot = len(df_snapshot)

    zero_percentage = (zero_count / total_snapshot) * 100
    none_percentage = (none_count / total_snapshot) * 100

    print(f"Income analysis: {zero_percentage:.2f}% zeros, {none_percentage:.2f}% missing")

    # Handle invalid income entries (missing values only, keep zeros)
    # DROP rows with missing income and sample additional valid rows to maintain 15,000 count
    if none_count > 0:
        print(f"\nDropping {none_count} invalid income entries (missing values)...")

        # Drop rows with missing income values (keep zeros)
        df_valid = df_snapshot.dropna(subset=['MonthlyIncome'])

        # Count how many additional valid rows we need to sample
        additional_needed = snapshot_size - len(df_valid)

        if additional_needed > 0:
            print(f"Need {additional_needed} additional valid rows to maintain 15,000 total...")

            # Get additional valid rows from the original dataset (excluding the ones already sampled)
            # First, identify the index of rows already in our sample
            sampled_indices = set(df_snapshot.index)

            # Filter original dataset to get rows not already sampled and with valid income
            df_remaining = df[~df.index.isin(sampled_indices)]
            df_remaining_valid = df_remaining.dropna(subset=['MonthlyIncome'])

            if len(df_remaining_valid) >= additional_needed:
                # Sample the needed number of additional valid rows
                additional_rows = df_remaining_valid.sample(n=additional_needed, random_state=42)

                # Combine with our valid rows
                df_snapshot = pd.concat([df_valid, additional_rows])

                print(f"Successfully added {len(additional_rows)} valid rows")
            else:
                print(f"Warning: Only {len(df_remaining_valid)} additional valid rows available, but need {additional_needed}")
                # Use what we have
                df_snapshot = pd.concat([df_valid, df_remaining_valid])

        else:
            # We already have enough valid rows
            df_snapshot = df_valid

        # Verify the result
        final_zero_count = (df_snapshot['MonthlyIncome'] == 0).sum()
        final_none_count = df_snapshot['MonthlyIncome'].isna().sum()
        final_total = len(df_snapshot)

        final_zero_percentage = (final_zero_count / final_total) * 100
        final_none_percentage = (final_none_count / final_total) * 100

        print(f"After processing: {final_total} total rows")
        print(f"Final analysis: {final_zero_percentage:.2f}% zeros, {final_none_percentage:.2f}% missing")
    else:
        print("No invalid income entries found")

    # Save the processed snapshot
    df_snapshot.to_csv(output_path, index=False)

    print(f"\nProcessed snapshot saved to: {output_path}")
    print(f"Snapshot contains {len(df_snapshot)} rows (maintained at 15,000)")

    return output_path

if __name__ == "__main__":
    extract_csv_snapshot()