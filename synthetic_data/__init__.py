# Synthetic Data Package
# This makes the synthetic_data directory a proper Python package

from .product_taxonomy import PRODUCT_TAXONOMY, PRODUCT_TYPES
from .balance_generator import generate_balances_for_product, generate_balances_for_multiple_products
from .eir_generator import generate_eir_for_product, generate_eir_for_multiple_products
from .lifetime_generator import generate_lifetime_for_product, generate_lifetime_for_multiple_products

__all__ = [
    'PRODUCT_TAXONOMY',
    'PRODUCT_TYPES',
    'generate_balances_for_product',
    'generate_balances_for_multiple_products',
    'generate_eir_for_product',
    'generate_eir_for_multiple_products',
    'generate_lifetime_for_product',
    'generate_lifetime_for_multiple_products'
]