# Dataset

- Source: German-credit-style educational data, `data/raw/credit_customers.csv`
  (copy of the repo's `credit_customers (DS).csv`).
- 1000 rows × 21 columns. Target `class`: good=700, bad=300.
- 7 numeric + 13 categorical features + target. No missing values, no duplicates.
- Split: stratified 80/20, `random_state=42` (`src/credit_risk.py::_split`).
- Preprocessing: `StandardScaler` (numeric) + `OneHotEncoder(handle_unknown=ignore)`
  (categorical) in a single `ColumnTransformer` used for train and inference.
- Limitations: synthetic/educational — do NOT claim real bank data.
