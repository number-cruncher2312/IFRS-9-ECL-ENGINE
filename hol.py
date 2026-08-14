import pandas as pd
import numpy as np

PATH = "outputs/full_10000_loan_portfolio.csv"

PRODUCT_BOUNDS = {
    "credit_card": {"balance_min": 500, "balance_max": 30_000, "eir_min": 0.12, "eir_max": 0.25, "lifetime_min": 1, "lifetime_max": 60, "lgd": 0.55},
    "auto_loan": {"balance_min": 15_000, "balance_max": 80_000, "eir_min": 0.04, "eir_max": 0.09, "lifetime_min": 12, "lifetime_max": 72, "lgd": 0.25},
    "mortgage": {"balance_min": 150_000, "balance_max": 1_500_000, "eir_min": 0.045, "eir_max": 0.075, "lifetime_min": 60, "lifetime_max": 360, "lgd": 0.10},
}

ACTIVE_PRODUCTS = {
    "credit_card": 0.7482,
    "auto_loan": 0.1280,
    "mortgage": 0.1238,
}


def fail(message):
    raise AssertionError(message)


print(f"Loading {PATH}...")
df = pd.read_csv(PATH)

print("\n=== BASIC ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Unique loan IDs: {df['loan_id'].nunique():,}")

if len(df) != 10_000:
    fail(f"Expected 10,000 rows, found {len(df):,}")

if df["loan_id"].nunique() != len(df):
    fail("Duplicate loan_id values found")

print("\n=== REQUIRED COLUMNS ===")
required = [
    "loan_id",
    "pd_origin",
    "pd_current",
    "product_type",
    "security_status",
    "collateral_type",
    "balance",
    "ead",
    "eir",
    "remaining_lifetime_months",
    "lgd_category",
    "lgd",
    "default_status",
]

missing = [c for c in required if c not in df.columns]
if missing:
    fail(f"Missing columns: {missing}")

print("All required columns present.")

print("\n=== PRODUCT MIX ===")
counts = df["product_type"].value_counts()
for product, target in ACTIVE_PRODUCTS.items():
    actual = counts.get(product, 0) / len(df)
    print(f"{product:12s}: {counts.get(product, 0):5d} ({actual:.2%}) | target {target:.2%}")

unexpected = set(df["product_type"]) - set(ACTIVE_PRODUCTS)
if unexpected:
    fail(f"Unexpected products found: {unexpected}")

print("\n=== PD ===")
for col in ["pd_origin", "pd_current"]:
    if not df[col].between(0, 1).all():
        fail(f"{col} contains values outside [0,1]")
    print(f"{col}: min={df[col].min():.6f}, mean={df[col].mean():.6f}, max={df[col].max():.6f}")

expected_defaults = df["pd_current"].sum()
actual_defaults = df["default_status"].sum()

print(f"Expected defaults (sum PD_current): {expected_defaults:.2f}")
print(f"Actual defaults:                    {actual_defaults:.0f}")
print(f"Expected default rate:              {expected_defaults / len(df):.4%}")
print(f"Actual default rate:                {actual_defaults / len(df):.4%}")

print("\n=== DEFAULT STATUS ===")
if not df["default_status"].isin([0, 1]).all():
    fail("default_status contains values other than 0/1")
print("Binary default status: OK")

print("\n=== EAD ===")
if not np.allclose(df["balance"], df["ead"]):
    fail("EAD is not equal to balance for every loan")
print("EAD = balance: OK")

print("\n=== PRODUCT-SPECIFIC VALIDATION ===")
for product, cfg in PRODUCT_BOUNDS.items():
    subset = df[df["product_type"] == product]

    if subset.empty:
        fail(f"No loans found for {product}")

    balance_ok = subset["balance"].between(
        cfg["balance_min"], cfg["balance_max"]
    ).all()

    eir_ok = subset["eir"].between(
        cfg["eir_min"], cfg["eir_max"]
    ).all()

    lifetime_ok = subset["remaining_lifetime_months"].between(
        cfg["lifetime_min"], cfg["lifetime_max"]
    ).all()

    lgd_ok = np.allclose(subset["lgd"], cfg["lgd"])

    print(f"\n{product}")
    print(f"  Count:      {len(subset):,}")
    print(f"  Balance:    {subset['balance'].min():,.2f} - {subset['balance'].max():,.2f} | {'OK' if balance_ok else 'FAIL'}")
    print(f"  EIR:        {subset['eir'].min():.4%} - {subset['eir'].max():.4%} | {'OK' if eir_ok else 'FAIL'}")
    print(f"  Lifetime:   {subset['remaining_lifetime_months'].min():.0f} - {subset['remaining_lifetime_months'].max():.0f} months | {'OK' if lifetime_ok else 'FAIL'}")
    print(f"  LGD:        {sorted(subset['lgd'].unique())} | {'OK' if lgd_ok else 'FAIL'}")

    if not balance_ok:
        fail(f"{product}: balance outside configured bounds")
    if not eir_ok:
        fail(f"{product}: EIR outside configured bounds")
    if not lifetime_ok:
        fail(f"{product}: lifetime outside configured bounds")
    if not lgd_ok:
        fail(f"{product}: incorrect LGD")

print("\n=== SECURITY / COLLATERAL ===")

expected_security = {
    "credit_card": ("unsecured", None),
    "auto_loan": ("secured", "vehicle"),
    "mortgage": ("secured", "real_estate"),
}

for product, (expected_sec, expected_col) in expected_security.items():
    subset = df[df["product_type"] == product]

    if not (subset["security_status"] == expected_sec).all():
        fail(f"{product}: incorrect security_status")

    actual_collateral = subset["collateral_type"].dropna().unique()

    if expected_col is None:
        if subset["collateral_type"].notna().any():
            fail(f"{product}: unsecured product has collateral")
    else:
        if not (subset["collateral_type"] == expected_col).all():
            fail(f"{product}: incorrect collateral_type")

    print(f"{product}: security/collateral OK")

print("\n=== EIR STATISTICS ===")
print(df.groupby("product_type")["eir"].agg(["count", "min", "median", "mean", "max", "std"]))

print("\n=== LIFETIME STATISTICS ===")
print(df.groupby("product_type")["remaining_lifetime_months"].agg(
    ["count", "min", "median", "mean", "max", "std"]
))

print("\n=== BALANCE STATISTICS ===")
print(df.groupby("product_type")["balance"].agg(
    ["count", "min", "median", "mean", "max", "std"]
))

print("\n=== FINAL RESULT ===")
print("ALL PHASE 1 PORTFOLIO CHECKS PASSED.")