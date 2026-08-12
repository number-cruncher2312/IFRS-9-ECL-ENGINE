# IFRS9 ECL Engine

This is a Python-based engine for calculating Expected Credit Loss (ECL) according to IFRS 9 standards.

## Project Structure

```
.
│
├── src/                 # ECL engine source code
│   ├── pd/              # Probability of Default models
│   ├── lgd/             # Loss Given Default models
│   ├── ead/             # Exposure at Default models
│   ├── staging/         # Data staging and preprocessing
│   └── ecl/             # ECL calculation logic
│
├── synthetic_data/      # Synthetic test data generation & modification
│   ├── modification.py          # Borrower snapshot modification module
│   ├── extract_snapshot.py      # Extract snapshots from source CSVs
│   └── credit_snapshot/         # Generated snapshot datasets
│
├── scripts/             # Standalone utility scripts
│   ├── process_gmsc_snapshot.py # Process GMSC CSV → current snapshot
│   ├── analyze_income.py        # Analyze zero/missing income ratios
│   ├── analyze_debt_ratio_zero_income.py  # DebtRatio analysis for zero-income
│   └── demo_zero_income.py      # Zero-income feature demonstration
│
├── tests/               # Unit and integration tests
│   ├── test_modification.py
│   └── test_zero_income.py
│
├── data/                # Source datasets
│   └── GiveMeSomeCredit-training.csv
│
├── outputs/             # Generated artifacts (plots, reports, etc.)
├── docs/                # Documentation
│   ├── CODE_REPORT.md
│   └── CODE_REPORT_CORRECTED.md
│
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate the credit snapshot (if not already present):
   ```bash
   python synthetic_data/extract_snapshot.py
   ```

3. Run tests:
   ```bash
   python -m pytest tests/
   ```

## Scripts

### Process GMSC snapshot
```bash
python scripts/process_gmsc_snapshot.py data/GiveMeSomeCredit-training.csv output_modified.csv [seed]
```

### Analyze income distribution
```bash
python scripts/analyze_income.py
```

### Analyze DebtRatio for zero-income borrowers
```bash
python scripts/analyze_debt_ratio_zero_income.py
```

### Zero-income demonstration
```bash
python scripts/demo_zero_income.py
```

## License

[Add license information here]
