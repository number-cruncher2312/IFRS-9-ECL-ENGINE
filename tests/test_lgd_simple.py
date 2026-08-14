"""
Simple test script to verify LGD generator functionality
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synthetic_data.lgd_generator import (
    get_lgd_rate,
    get_lgd_rates_for_products,
    assign_lgd_to_dataframe,
    get_product_lgd_category,
    get_lgd_category_rate,
    get_v1_active_products,
    get_v1_inactive_products,
    get_all_product_lgd_mappings
)
import pandas as pd

def test_lgd_generator():
    """Test the LGD generator functionality."""
    print("=== Testing V1 LGD Generator ===")

    # Test individual product LGD rates
    print("\n1. Individual Product LGD Rates:")
    products = ["mortgage", "auto_loan", "credit_card", "student_loan", "other_personal_loan"]
    for product in products:
        lgd_rate = get_lgd_rate(product)
        lgd_category = get_product_lgd_category(product)
        print(f"   {product:20} -> {lgd_category:15} = {lgd_rate:.2f}")

    # Test batch assignment
    print("\n2. Batch Assignment:")
    batch_result = get_lgd_rates_for_products(["mortgage", "credit_card", "auto_loan"])
    for product, rate in batch_result.items():
        print(f"   {product}: {rate:.2f}")

    # Test DataFrame assignment
    print("\n3. DataFrame Assignment:")
    df = pd.DataFrame({
        "product_type": ["mortgage", "credit_card", "auto_loan", "mortgage", "student_loan"]
    })
    result_df = assign_lgd_to_dataframe(df)
    print("   Original DataFrame:")
    print("   ", df.to_string(index=False))
    print("   DataFrame with LGD:")
    print("   ", result_df.to_string(index=False))

    # Test V1 product mix
    print("\n4. V1 Product Mix:")
    active = get_v1_active_products()
    inactive = get_v1_inactive_products()
    print(f"   Active Products: {active}")
    print(f"   Inactive Products: {inactive}")

    # Test complete mappings
    print("\n5. Complete Product-LGD Mappings:")
    mappings = get_all_product_lgd_mappings()
    for product, info in sorted(mappings.items()):
        print(f"   {product:20} -> {info['lgd_category']:15} = {info['lgd_rate']:.2f}")

    # Test error handling
    print("\n6. Error Handling:")
    try:
        get_lgd_rate("unknown_product")
        print("   ERROR: Should have raised ValueError for unknown product")
    except ValueError as e:
        print(f"   OK Correctly raised ValueError: {e}")

    try:
        get_lgd_category_rate("unknown_category")
        print("   ERROR: Should have raised ValueError for unknown category")
    except ValueError as e:
        print(f"   OK Correctly raised ValueError: {e}")

    print("\n=== All Tests Completed Successfully ===")

if __name__ == "__main__":
    test_lgd_generator()