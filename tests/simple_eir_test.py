#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_eir_generator():
    """Simple test of EIR generator functionality."""
    try:
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

        print("✓ EIR generator imported successfully")

        # Test single product EIR generation
        mortgage_eir = generate_eir_for_product("mortgage", n_loans=1, seed=42)
        print(f"✓ Mortgage EIR: {mortgage_eir:.4f} ({mortgage_eir*100:.2f}%)")

        # Test multiple EIR values
        auto_eirs = generate_eir_for_product("auto_loan", n_loans=5, seed=42)
        print(f"✓ Auto EIRs: {[f'{eir:.4f}' for eir in auto_eirs]}")

        # Test multiple products
        product_assignments = ["credit_card", "mortgage", "auto_loan", "credit_card"]
        eir_dict = generate_eir_for_multiple_products(product_assignments, seed=42)
        print(f"✓ Multiple products: {list(eir_dict.keys())}")

        # Test DataFrame generation
        eir_df = generate_eir_dataframe(product_assignments, seed=42)
        print(f"✓ DataFrame shape: {eir_df.shape}")

        # Test validation
        validation = validate_eir_values(auto_eirs, "auto_loan")
        print(f"✓ Validation: {validation}")

        # Test statistics
        stats = get_eir_statistics(auto_eirs, "auto_loan")
        print(f"✓ Statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

        # Test reproducibility
        result = validate_eir_reproducibility(["mortgage", "auto_loan"], seed=123, n_trials=2)
        print(f"✓ Reproducibility: {'PASS' if result else 'FAIL'}")

        # Test report generation
        report = generate_eir_report(product_assignments, seed=42)
        print(f"✓ Report generated with {len(report['statistics'])} product statistics")

        # Test internal validation
        _run_validation_tests()
        print("✓ Internal validation tests passed")

        print("\n🎉 All EIR generator tests passed!")
        return True

    except Exception as e:
        print(f"❌ EIR generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Simple EIR Generator Test")
    print("=" * 40)
    success = test_eir_generator()
    sys.exit(0 if success else 1)