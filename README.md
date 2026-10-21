# Credit Card Fraud Analysis (end-to-end)

Statistical + machine-learning analysis of 284,807 card transactions, built as a
portfolio project for an RBI DSIM-style (statistics & information management) profile.

## Run it
1. Download `creditcard.csv` from Kaggle ("Credit Card Fraud Detection", ULB / Worldline)
   and place it at `data/creditcard.csv`.
2. `pip install -r requirements.txt`
3. `python fraud_pipeline.py --data data/creditcard.csv --out outputs`
   (about 2.5 minutes on a single CPU core; fully reproducible, seed = 42)

## Outputs
- `outputs/results.json`   every number quoted in REPORT.md
- `outputs/figures/`       8 charts
- `outputs/tables/`        model comparison, alert-budget table, feature importance, drift (PSI)
- `outputs/model/fraud_model.joblib`   trained model + chosen threshold + feature list

## Read next
`REPORT.md` explains what was done and why, the business findings, limitations and interview Q&A.
