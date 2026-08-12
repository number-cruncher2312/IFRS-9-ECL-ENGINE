"""
V1 Synthetic Product/Security Taxonomy
======================================

Static lookup defining product types, security/collateral status, and
synthetic assumptions used consistently for:
  - LGD assignment
  - balance / EAD generation
  - EIR assumptions
  - remaining lifetime

IMPORTANT: All balance, EIR, lifetime, and LGD numbers in this file are
SYNTHETIC V1 ASSUMPTIONS for wiring only. They are NOT empirical values,
NOT calibrated to GMSC or any real portfolio, and should not be used as
if they were market-observed parameters.

Design notes
------------
- `product_type`, `security_status`, and `collateral_type` are kept as
  SEPARATE fields so that, in principle, a product could be either
  secured or unsecured even though the V1 defaults below are
  product-consistent.
- `collateral_type` is only meaningful when `security_status == "secured"`
  and must be `None` when unsecured.
- One LGD category per product (no subcategories in V1).
- Balance ranges are generation priors (sampling bounds), not hard caps.
  Balances are USD-denominated for V1.
- `remaining_lifetime_months` is contractual remaining tenor at the
  snapshot date (months), not original tenor.

V1 product set
--------------
The V1 taxonomy models exactly five retail-lending products:
  - mortgage
  - auto_loan
  - credit_card
  - student_loan
  - other_personal_loan

HELOC note
----------
Home-equity lines of credit (HELOC) are NOT a separate product in V1;
they are merged into `mortgage`. The `mortgage` product therefore
represents real-estate-secured lending (first-lien mortgages and HELOCs)
rather than strictly closed-end first mortgages only. This keeps the
product set small for V1 wiring while preserving the real-estate-secured
economic signature (low LGD, long lifetime, large balances).

Credit-card EAD note
--------------------
For V1, credit-card exposure is simplified:
  EAD = current outstanding balance
i.e. the exposure is treated as fully drawn. There is NO unused-limit
model and NO CCF (credit-conversion-factor) adjustment in V1. This is a
deliberate simplification and should be revisited before any
calibration-grade use.
"""

from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Product type enum (string constants keep the rest of the code stable)
# ---------------------------------------------------------------------------
PRODUCT_TYPES = {
    "mortgage",
    "auto_loan",
    "credit_card",
    "student_loan",
    "other_personal_loan",
}


# ---------------------------------------------------------------------------
# Security status enum
# ---------------------------------------------------------------------------
SECURITY_STATUS = {
    "secured",
    "unsecured",
}


# ---------------------------------------------------------------------------
# LGD categories (one per product in V1)
# ---------------------------------------------------------------------------
LGD_CATEGORIES = {
    "LGD_MORTGAGE",
    "LGD_AUTO",
    "LGD_CREDIT_CARD",
    "LGD_STUDENT",
    "LGD_UNSECURED",
}


# ---------------------------------------------------------------------------
# Core taxonomy: per-product synthetic V1 assumptions
#
# Valid product/security/collateral mappings:
#   mortgage            -> secured   -> real_estate
#   auto_loan           -> secured   -> vehicle
#   credit_card         -> unsecured -> None
#   student_loan        -> unsecured -> None
#   other_personal_loan -> unsecured -> None
#
# Fields:
#   security_status          : "secured" | "unsecured"
#   collateral_type          : string, or None when unsecured
#   balance                  : generation prior in USD
#       - min, max, median   (lognormal/truncated sampling bounds)
#   eir                      : effective interest rate (annual, decimal)
#       - min, max, base
#   remaining_lifetime_months: contractual remaining tenor at snapshot (months)
#       - min, max, median
#   lgd_category             : maps to LGD_RATES below
#
# All numeric values below are SYNTHETIC V1 ASSUMPTIONS, not empirical.
# ---------------------------------------------------------------------------
PRODUCT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "mortgage": {
        # HELOC merged in here for V1: represents real-estate-secured
        # lending, not strictly closed-end first mortgages.
        "security_status": "secured",
        "collateral_type": "real_estate",
        "balance": {
            "min": 150_000,
            "max": 1_500_000,
            "median": 350_000,
        },
        "eir": {
            "min": 0.045,
            "max": 0.075,
            "base": 0.060,
        },
        "remaining_lifetime_months": {
            "min": 60,
            "max": 360,
            "median": 180,
        },
        "lgd_category": "LGD_MORTGAGE",
    },
    "auto_loan": {
        "security_status": "secured",
        "collateral_type": "vehicle",
        "balance": {
            "min": 15_000,
            "max": 80_000,
            "median": 32_000,
        },
        "eir": {
            "min": 0.040,
            "max": 0.090,
            "base": 0.065,
        },
        "remaining_lifetime_months": {
            "min": 12,
            "max": 72,
            "median": 48,
        },
        "lgd_category": "LGD_AUTO",
    },
    "credit_card": {
        # V1 simplification: EAD = current outstanding balance
        # (fully drawn; no unused-limit / CCF model).
        "security_status": "unsecured",
        "collateral_type": None,
        "balance": {
            "min": 500,
            "max": 30_000,
            "median": 6_000,
        },
        "eir": {
            "min": 0.12,
            "max": 0.25,
            "base": 0.19,
        },
        "remaining_lifetime_months": {
            "min": 1,
            "max": 60,
            "median": 18,
        },
        "lgd_category": "LGD_CREDIT_CARD",
    },
    "student_loan": {
        "security_status": "unsecured",
        "collateral_type": None,
        "balance": {
            "min": 5_000,
            "max": 120_000,
            "median": 35_000,
        },
        "eir": {
            "min": 0.040,
            "max": 0.080,
            "base": 0.055,
        },
        "remaining_lifetime_months": {
            "min": 12,
            "max": 240,
            "median": 120,
        },
        "lgd_category": "LGD_STUDENT",
    },
    "other_personal_loan": {
        "security_status": "unsecured",
        "collateral_type": None,
        "balance": {
            "min": 1_000,
            "max": 60_000,
            "median": 15_000,
        },
        "eir": {
            "min": 0.08,
            "max": 0.22,
            "base": 0.14,
        },
        "remaining_lifetime_months": {
            "min": 6,
            "max": 84,
            "median": 36,
        },
        "lgd_category": "LGD_UNSECURED",
    },
}


# ---------------------------------------------------------------------------
# LGD category -> synthetic V1 LGD rate
#
# These are SYNTHETIC V1 ASSUMPTIONS, not calibrated/empirical values.
# ---------------------------------------------------------------------------
LGD_RATES: Dict[str, float] = {
    "LGD_MORTGAGE": 0.10,        # mortgage (real-estate-secured)
    "LGD_AUTO": 0.25,            # auto loan (vehicle-secured)
    "LGD_CREDIT_CARD": 0.55,     # credit card (unsecured, revolver)
    "LGD_STUDENT": 0.20,         # student loan (unsecured, high recovery)
    "LGD_UNSECURED": 0.45,       # other personal loan (unsecured)
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def get_lgd_rate(product_type: str) -> float:
    """Return the synthetic V1 LGD rate for a product type."""
    if product_type not in PRODUCT_TAXONOMY:
        raise ValueError(
            f"Unknown product_type: {product_type!r}. "
            f"Valid types: {sorted(PRODUCT_TYPES)}"
        )
    return LGD_RATES[PRODUCT_TAXONOMY[product_type]["lgd_category"]]


def get_product_config(product_type: str) -> Dict[str, Any]:
    """Return the full V1 config block for a product type."""
    if product_type not in PRODUCT_TAXONOMY:
        raise ValueError(
            f"Unknown product_type: {product_type!r}. "
            f"Valid types: {sorted(PRODUCT_TYPES)}"
        )
    return PRODUCT_TAXONOMY[product_type]


# ---------------------------------------------------------------------------
# Internal validation tests (lightweight, self-contained)
#
# Runs when this module is executed directly (`python product_taxonomy.py`)
# and validates the V1 taxonomy invariants. Does NOT test product/portfolio
# generation (out of scope for this file).
# ---------------------------------------------------------------------------
_VALID_SECURITY_FOR_COLLATERAL = {
    "real_estate": "secured",
    "vehicle": "secured",
    None: "unsecured",
}


def _run_validation_tests() -> None:
    """Validate the V1 taxonomy invariants. Raises AssertionError on failure."""
    # 1. All 5 expected products exist.
    expected = {
        "mortgage",
        "auto_loan",
        "credit_card",
        "student_loan",
        "other_personal_loan",
    }
    assert set(PRODUCT_TYPES) == expected, (
        f"PRODUCT_TYPES mismatch: {set(PRODUCT_TYPES)} != {expected}"
    )
    assert set(PRODUCT_TAXONOMY.keys()) == expected, (
        f"PRODUCT_TAXONOMY keys mismatch: "
        f"{set(PRODUCT_TAXONOMY.keys())} != {expected}"
    )

    # 2. Each product has a valid security/collateral mapping consistent
    #    with the documented product->security->collateral table.
    expected_mapping = {
        "mortgage": ("secured", "real_estate"),
        "auto_loan": ("secured", "vehicle"),
        "credit_card": ("unsecured", None),
        "student_loan": ("unsecured", None),
        "other_personal_loan": ("unsecured", None),
    }
    for product, (sec, col) in expected_mapping.items():
        cfg = PRODUCT_TAXONOMY[product]
        assert cfg["security_status"] == sec, (
            f"{product}: security_status={cfg['security_status']!r}, "
            f"expected {sec!r}"
        )
        assert cfg["collateral_type"] is col or cfg["collateral_type"] == col, (
            f"{product}: collateral_type={cfg['collateral_type']!r}, "
            f"expected {col!r}"
        )
        assert cfg["security_status"] in SECURITY_STATUS, (
            f"{product}: security_status {cfg['security_status']!r} "
            f"not in SECURITY_STATUS"
        )
        assert _VALID_SECURITY_FOR_COLLATERAL.get(cfg["collateral_type"]) == \
            cfg["security_status"], (
            f"{product}: collateral_type {cfg['collateral_type']!r} "
            f"inconsistent with security_status "
            f"{cfg['security_status']!r}"
        )

    # 3. Every product maps to an LGD category, and that category is known.
    for product, cfg in PRODUCT_TAXONOMY.items():
        cat = cfg["lgd_category"]
        assert cat in LGD_CATEGORIES, (
            f"{product}: lgd_category {cat!r} not in LGD_CATEGORIES"
        )

    # 4. Every LGD category has a rate, and every rate maps to a category.
    assert set(LGD_RATES.keys()) == LGD_CATEGORIES, (
        f"LGD_RATES keys {set(LGD_RATES.keys())} != "
        f"LGD_CATEGORIES {LGD_CATEGORIES}"
    )
    for cat, rate in LGD_RATES.items():
        assert isinstance(rate, float), (
            f"{cat}: LGD rate is not a float ({type(rate)})"
        )
        assert 0.0 <= rate <= 1.0, (
            f"{cat}: LGD rate {rate} out of [0, 1]"
        )

    # 5. Helper functions work for every product and reject unknowns.
    for product in PRODUCT_TYPES:
        rate = get_lgd_rate(product)
        assert rate == LGD_RATES[PRODUCT_TAXONOMY[product]["lgd_category"]], (
            f"{product}: get_lgd_rate returned {rate}, expected "
            f"{LGD_RATES[PRODUCT_TAXONOMY[product]['lgd_category']]}"
        )
        cfg = get_product_config(product)
        assert cfg is PRODUCT_TAXONOMY[product], (
            f"{product}: get_product_config did not return the taxonomy block"
        )

    try:
        get_lgd_rate("nonexistent_product")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "get_lgd_rate did not raise ValueError for unknown product"
        )

    try:
        get_product_config("nonexistent_product")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "get_product_config did not raise ValueError for unknown product"
        )


if __name__ == "__main__":
    _run_validation_tests()
    print("product_taxonomy V1 validation: OK")
    print(f"  products: {sorted(PRODUCT_TYPES)}")
    print(f"  lgd_categories: {sorted(LGD_CATEGORIES)}")
    print(f"  lgd_rates: {LGD_RATES}")
