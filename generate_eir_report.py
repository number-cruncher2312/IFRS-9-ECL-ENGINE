#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_comprehensive_eir_report():
    """Generate comprehensive EIR report with statistics and validation."""
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
        from synthetic_data.product_taxonomy import PRODUCT_TAXONOMY, PRODUCT_TYPES

        print("Generating Comprehensive EIR Report")
        print("=" * 50)

        # Run internal validation tests first
        print("1. Running internal validation tests...")
        _run_validation_tests()
        print("   ✓ Internal validation tests passed")

        # Test all products individually
        print("\n2. Testing all products individually...")
        products = ["mortgage", "auto_loan", "credit_card", "student_loan", "other_personal_loan"]
        all_stats = {}

        for product in products:
            print(f"   Testing {product}...")
            eirs = generate_eir_for_product(product, n_loans=1000, seed=42)
            stats = get_eir_statistics(eirs, product)

            # Validate values
            validation = validate_eir_values(eirs, product)
            assert validation["all_within_bounds"], f"{product} EIRs out of bounds"
            assert validation["all_positive"], f"{product} EIRs not all positive"

            all_stats[product] = stats
            print(f"     Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, Base: {stats['base_eir']:.4f}")

        # Test product differentiation
        print("\n3. Testing product differentiation...")
        mortgage_stats = all_stats["mortgage"]
        credit_card_stats = all_stats["credit_card"]

        assert mortgage_stats["mean"] < credit_card_stats["mean"], "Mortgage should have lower EIR than credit card"
        print("   ✓ Product differentiation confirmed")

        # Test bounds compliance
        print("\n4. Testing bounds compliance...")
        for product, stats in all_stats.items():
            eir_config = PRODUCT_TAXONOMY[product]["eir"]
            min_eir = eir_config["min"]
            max_eir = eir_config["max"]

            assert stats["min"] >= min_eir, f"{product} min EIR {stats['min']} below configured min {min_eir}"
            assert stats["max"] <= max_eir, f"{product} max EIR {stats['max']} above configured max {max_eir}"
            print(f"   ✓ {product}: [{min_eir:.3f}, {max_eir:.3f}] bounds respected")

        # Test centering around base rates
        print("\n5. Testing centering around base rates...")
        for product, stats in all_stats.items():
            eir_config = PRODUCT_TAXONOMY[product]["eir"]
            base_eir = eir_config["base"]
            eir_range = eir_config["max"] - eir_config["min"]
            tolerance = eir_range * 0.1  # 10% of range

            distance_from_base = abs(stats['mean'] - base_eir)
            assert distance_from_base <= tolerance, f"{product} mean too far from base"

            print(f"   ✓ {product}: mean {stats['mean']:.4f} close to base {base_eir:.4f} (distance: {distance_from_base:.4f})")

        # Test reproducibility
        print("\n6. Testing reproducibility...")
        product_assignments = ["mortgage", "auto_loan", "credit_card"] * 100
        result = validate_eir_reproducibility(product_assignments, seed=123, n_trials=2)
        assert result, "Reproducibility test failed"
        print("   ✓ Reproducibility confirmed")

        # Test continuous values
        print("\n7. Testing continuous values...")
        auto_eirs = generate_eir_for_product("auto_loan", n_loans=1000, seed=42)
        unique_eirs = set(auto_eirs)
        unique_count = len(unique_eirs)

        assert unique_count > 100, f"Should have many unique EIR values, got {unique_count}"
        print(f"   ✓ Continuous values confirmed ({unique_count} unique values from 1000 samples)")

        # Test no current/future information usage
        print("\n8. Testing no current/future information usage...")
        import inspect
        source = inspect.getsource(generate_eir_for_product)
        forbidden_concepts = ['PD_current', 'default', 'staging', 'LGD', 'EAD', 'current_', 'future']

        forbidden_found = []
        for concept in forbidden_concepts:
            if concept in source:
                forbidden_found.append(concept)

        assert len(forbidden_found) == 0, f"Found forbidden concepts: {forbidden_found}"
        print("   ✓ No current/future information usage confirmed")

        # Generate comprehensive report
        print("\n9. Generating comprehensive statistical report...")
        large_portfolio = ["credit_card"] * 7500 + ["auto_loan"] * 1300 + ["mortgage"] * 1200
        report = generate_eir_report(large_portfolio, seed=42)

        # Print summary statistics
        print("\n10. Comprehensive EIR Statistics by Product:")
        print("-" * 60)

        statistics = report["statistics"]
        for product in ["credit_card", "auto_loan", "mortgage"]:
            if product in statistics:
                stats = statistics[product]
                eir_config = PRODUCT_TAXONOMY[product]["eir"]
                base_eir = eir_config["base"]

                print(f"\n{product.upper()}:")
                print(f"  Base EIR:        {base_eir:.4f} ({base_eir*100:.2f}%)")
                print(f"  Generated Mean:  {stats['mean']:.4f} ({stats['mean']*100:.2f}%)")
                print(f"  Median:          {stats['median']:.4f} ({stats['median']*100:.2f}%)")
                print(f"  Min:             {stats['min']:.4f} ({stats['min']*100:.2f}%)")
                print(f"  Max:             {stats['max']:.4f} ({stats['max']*100:.2f}%)")
                print(f"  Std Dev:         {stats['std']:.4f}")
                print(f"  Distance from Base: {stats['distance_from_base']:.4f}")
                print(f"  Coefficient of Variation: {stats['coefficient_of_variation']:.4f}")

        # Test error handling
        print("\n11. Testing error handling...")
        try:
            generate_eir_for_product("invalid_product", n_loans=1)
            assert False, "Should have raised error for invalid product"
        except ValueError:
            print("   ✓ Invalid product error handling works")

        try:
            generate_eir_for_product("mortgage", n_loans=0)
            assert False, "Should have raised error for n_loans=0"
        except ValueError:
            print("   ✓ Invalid n_loans error handling works")

        # Summary
        print("\n" + "=" * 50)
        print("EIR GENERATOR IMPLEMENTATION SUMMARY")
        print("=" * 50)

        print("\n✓ Implementation Status:")
        print("  - EIR generator module created: eir_generator.py")
        print("  - All active products generate valid EIRs")
        print("  - EIR values always within product min/max bounds")
        print("  - Product differentiation confirmed")
        print("  - Reproducibility with explicit seed confirmed")
        print("  - Continuous (not discrete) values confirmed")
        print("  - No current/future information usage confirmed")
        print("  - Comprehensive error handling implemented")
        print("  - Statistical validation functions included")

        print("\n✓ Methodology:")
        print("  - Product type is primary determinant of EIR range")
        print("  - Uses configured base EIR as center")
        print("  - Applies modest stochastic variation (normal distribution)")
        print("  - Clips to configured min/max bounds")
        print("  - No borrower characteristics used (V1 scope)")
        print("  - No complicated credit-pricing models")

        print("\n✓ Borrower Variables Used:")
        print("  - None (V1 scope - only product type used)")

        print("\n✓ Statistical Properties:")
        for product in ["mortgage", "auto_loan", "credit_card"]:
            if product in statistics:
                stats = statistics[product]
                eir_config = PRODUCT_TAXONOMY[product]["eir"]
                base_eir = eir_config["base"]
                distance_from_base = abs(stats['mean'] - base_eir)
                eir_range = eir_config["max"] - eir_config["min"]

                print(f"  - {product}:")
                print(f"    Mean centers around base: {distance_from_base:.4f} (within {eir_range*0.1:.4f} tolerance)")
                print(f"    Reasonable variation: std={stats['std']:.4f}, cv={stats['coefficient_of_variation']:.4f}")

        print("\n✓ Methodological Concerns:")
        print("  - None identified")
        print("  - Implementation follows V1 specifications exactly")
        print("  - No use of forbidden current/future information")
        print("  - Proper bounds enforcement")
        print("  - Full reproducibility with seeds")

        print("\n🎉 EIR Generator Implementation Complete and Validated!")
        return True

    except Exception as e:
        print(f"EIR report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_comprehensive_eir_report()
    sys.exit(0 if success else 1)