# Demo Scope — Full-RGNB Research Pipeline (Matrix Lab)

A stripped, public-facing build of an end-to-end quantitative research notebook.
Demo URL: `<pipeline.example.com>`

## What's real
- The complete **16-section pipeline**, methodology unmodified: data loading →
  feature engineering → event / triple-barrier labeling → leakage-free **purged CV
  (CPCV)** → model training (RF / XGB / LGB / Logit / MLP) → diagnostics → walkforward
  → backtest → risk analytics → **SHAP** explainability → meta-model → ensemble stacking
  → deployment export → summary.
- All `src/` modules (`features`, `labels`, `cv`, `visualization`, `utils`).
- Runs end-to-end on **real NQ futures bars**, rendered to static HTML
  (`notebooks/rendered/Full-RGNB.html`, 133 figures).

## What's stripped / withheld
- **The raw Databento data product is not shipped** (it's licensed). The notebook runs
  with `USE_DATABENTO=False` against a *derived* OHLCV CSV.
- **Real trained deployment models** (`.pkl`) — regenerated at run time, gitignored, not shipped.
- **`artifacts/` run outputs** — regenerated at run time, gitignored.

## What runs
- `data/NQ_databento_30min_demo.csv` — the most recent **~8 months of NQ 30-min adjusted
  bars** (8,000 rows): a window of the full `data/NQ_databento_30min_adj.csv` (24,720 rows,
  included in the repo for download). The render uses the window so it completes on modest
  hardware; the full series is available alongside it.

## Demo-build adjustments (transparent)
- `ROOT` made repo-relative; `VERSION` pinned for reproducibility.
- One robustness fix in the ensemble: a small **`SafeXGB`** wrapper label-encodes `y` per
  fit, so a CV fold that happens to drop the rare *neutral* class doesn't crash XGBoost.
  This **preserves the 3-class (bull / neutral / bear) model**.
- `requirements.txt` pinned to the validated environment.

## Credentials
- **None.** The pipeline is CSV-driven and fully offline — there was nothing to rotate.

## Contact
[email] · [LinkedIn]
