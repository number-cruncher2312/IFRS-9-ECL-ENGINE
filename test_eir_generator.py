#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synthetic_data.eir_generator import (
    generate_eir_for_product,
    generate_eir_for_multiple_products,
    generate_eir_dataframe,
    validate_eir_values,
    get_eir_statistics,
    validate_eir_reproducibility,
    generate_eir_report,
    _run_validation_tests
)

def test_basic_functionality():
    """Test basic EIR generator functionality."""
    print("Testing basic EIR generator functionality...")

    # Test single product EIR generation
    print("1. Testing single product EIR generation...")
    mortgage_eir = generate_eir_for_product("mortgage", n_loans=1, seed=42)
    print(f"   Mortgage EIR: {mortgage_eir:.4f} ({mortgage_eir*100:.2f}%)")
    assert 0.045 <= mortgage_eir <= 0.075, f"Mortgage EIR {mortgage_eir} out of bounds"

    # Test multiple EIR values
    print("2. Testing multiple EIR values...")
    auto_eirs = generate_eir_for_product("auto_loan", n_loans=5, seed=42)
    print(f"   Auto EIRs: {[f'{eir:.4f}' for eir in auto_eirs]}")
    assert all(0.04 <= eir <= 0.09 for eir in auto_eirs), "Auto EIRs out of bounds"

    # Test multiple products
    print("3. Testing multiple products...")
    product_assignments = ["credit_card", "mortgage", "auto_loan", "credit_card", "student_loan"]
    eir_dict = generate_eir_for_multiple_products(product_assignments, seed=42)
    print(f"   Products: {list(eir_dict.keys())}")
    print(f"   Counts: {[len(eirs) for eirs in eir_dict.values()]}")

    # Test DataFrame generation
    print("4. Testing DataFrame generation...")
    eir_df = generate_eir_dataframe(product_assignments, seed=42)
    print(f"   DataFrame shape: {eir_df.shape}")
    print(f"   Columns: {list(eir_df.columns)}")
    print(f"   First few rows:\n{eir_df.head()}")

    # Test validation
    print("5. Testing validation...")
    validation = validate_eir_values(auto_eirs, "auto_loan")
    print(f"   Validation results: {validation}")

    # Test statistics
    print("6. Testing statistics...")
    stats = get_eir_statistics(auto_eirs, "auto_loan")
    print(f"   Statistics: {stats}")

    # Test reproducibility
    print("7. Testing reproducibility...")
    result = validate_eir_reproducibility(["mortgage", "auto_loan"], seed=123, n_trials=2)
    print(f"   Reproducibility test: {'PASS' if result else 'FAIL'}")

    # Test report generation
    print("8. Testing report generation...")
    report = generate_eir_report(product_assignments, seed=42)
    print(f"   Report keys: {list(report.keys())}")
    print(f"   Products in report: {list(report['statistics'].keys())}")

    print("All basic tests passed!")

def test_all_products():
    """Test EIR generation for all product types."""
    print("\nTesting EIR generation for all product types...")

    products = ["mortgage", "auto_loan", "credit_card", "student_loan", "other_personal_loan"]
    expected_ranges = {
        "mortgage": (0.045, 0.075),
        "auto_loan": (0.04, 0.09),
        "credit_card": (0.12, 0.25),
        "student_loan": (0.04, 0.08),
        "other_personal_loan": (0.08, 0.22)
    }

    for product in products:
        print(f"Testing {product}...")
        eirs = generate_eir_for_product(product, n_loans=10, seed=42)
        min_eir, max_eir = expected_ranges[product]

        # Check bounds
        assert all(min_eir <= eir <= max_eir for eir in eirs), f"{product} EIRs out of bounds"

        # Check statistics
        stats = get_eir_statistics(eirs, product)
        print(f"  {product}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, base={stats['base_eir']:.4f}")

    print("All product tests passed!")

def test_statistical_properties():
    """Test statistical properties of EIR generation."""
    print("\nTesting statistical properties...")

    # Define expected ranges for statistical testing
    expected_ranges = {
        "mortgage": (0.045, 0.075),
        "auto_loan": (0.04, 0.09),
        "credit_card": (0.12, 0.25),
        "student_loan": (0.04, 0.08),
        "other_personal_loan": (0.08, 0.22)
    }

    # Test that EIRs center around base rates
    products = ["mortgage", "auto_loan", "credit_card"]
    n_samples = 1000

    for product in products:
        print(f"Testing {product} statistical properties...")
        eirs = generate_eir_for_product(product, n_loans=n_samples, seed=42)
        stats = get_eir_statistics(eirs, product)

        # Check that mean is close to base (within 10% of range)
        base_eir = stats['base_eir']
        eir_range = expected_ranges[product][1] - expected_ranges[product][0]
        tolerance = eir_range * 0.1

        distance_from_base = abs(stats['mean'] - base_eir)
        print(f"  {product}: mean={stats['mean']:.4f}, base={base_eir:.4f}, distance={distance_from_base:.4f}, tolerance={tolerance:.4f}")

        assert distance_from_base <= tolerance, f"{product} mean too far from base"

    print("Statistical properties test passed!")

if __name__ == "__main__":
    print("EIR Generator Test Script")
    print("=" * 50)

    # Run internal validation tests
    print("Running internal validation tests...")
    _run_validation_tests()

    # Run our custom tests
    test_basic_functionality()
    test_all_products()
    test_statistical_properties()

    print("\n" + "=" * 50)
    print("All EIR generator tests completed successfully!")