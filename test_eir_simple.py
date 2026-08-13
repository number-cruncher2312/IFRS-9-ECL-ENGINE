#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        print("Testing EIR generator...")
        from synthetic_data.eir_generator import generate_eir_for_product

        # Test mortgage EIR
        mortgage_eir = generate_eir_for_product("mortgage", n_loans=1, seed=42)
        print(f"Mortgage EIR: {mortgage_eir:.4f} ({mortgage_eir*100:.2f}%)")

        # Test auto loan EIR
        auto_eir = generate_eir_for_product("auto_loan", n_loans=1, seed=42)
        print(f"Auto loan EIR: {auto_eir:.4f} ({auto_eir*100:.2f}%)")

        # Test credit card EIR
        credit_card_eir = generate_eir_for_product("credit_card", n_loans=1, seed=42)
        print(f"Credit card EIR: {credit_card_eir:.4f} ({credit_card_eir*100:.2f}%)")

        print("EIR generator test successful!")

    except Exception as e:
        print(f"EIR generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)