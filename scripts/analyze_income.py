import pandas as pd
import sys

def analyze_income_zeros(csv_file_path):
    """
    Analyze the percentage of borrowers with income field of 'none' or 'zero'

    Args:
        csv_file_path (str): Path to the CSV file containing credit data

    Returns:
        dict: Dictionary containing separate percentages for 'none' and 'zero' income
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Check if MonthlyIncome column exists
        if 'MonthlyIncome' not in df.columns:
            print("Error: MonthlyIncome column not found in the CSV file")
            return None

        # Count total borrowers
        total_borrowers = len(df)

        # Count borrowers with income = 0 (zero)
        zero_count = (df['MonthlyIncome'] == 0).sum()
        zero_percentage = (zero_count / total_borrowers) * 100

        # Count borrowers with missing income (NaN - considered 'none')
        none_count = df['MonthlyIncome'].isna().sum()
        none_percentage = (none_count / total_borrowers) * 100

        # Count combined borrowers with income = 0 or missing
        zero_or_none_count = zero_count + none_count
        combined_percentage = (zero_or_none_count / total_borrowers) * 100

        print(f"Total borrowers: {total_borrowers}")
        print(f"Borrowers with income = 0 (zero): {zero_count} ({zero_percentage:.2f}%)")
        print(f"Borrowers with missing income (none): {none_count} ({none_percentage:.2f}%)")
        print(f"Borrowers with income = 0 or missing: {zero_or_none_count} ({combined_percentage:.2f}%)")

        return {
            'total_borrowers': total_borrowers,
            'zero_count': zero_count,
            'zero_percentage': zero_percentage,
            'none_count': none_count,
            'none_percentage': none_percentage,
            'combined_count': zero_or_none_count,
            'combined_percentage': combined_percentage
        }

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error analyzing data: {str(e)}")
        return None

if __name__ == "__main__":
    # Default CSV file path (relative to project root, accounting for scripts/ location)
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_csv_path = os.path.join(project_root, "synthetic_data", "credit_snapshot", "GiveMeSomeCredit-snapshot.csv")

    # If command line argument is provided, use that instead
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_csv_path

    print(f"Analyzing income data from: {csv_path}")
    result = analyze_income_zeros(csv_path)

    if result is not None:
        print(f"\nFinal result:")
        print(f"- {result['zero_percentage']:.2f}% of borrowers have income = 'zero'")
        print(f"- {result['none_percentage']:.2f}% of borrowers have income = 'none' (missing)")
        print(f"- {result['combined_percentage']:.2f}% of borrowers have income = 'none' or 'zero'")
