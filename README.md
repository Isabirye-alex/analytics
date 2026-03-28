# Customer Analytics & Churn Intelligence Pipeline

A production-grade, modular customer analytics system built on transactional retail data. Transforms raw sales records into operational intelligence including customer segmentation, revenue analysis, churn prediction, and an interactive business dashboard.

---

## What It Does

| Capability | Description |
|---|---|
| **Churn Prediction** | GradientBoosting model — cross-validated F1 0.77, recall 0.81 |
| **RFM Segmentation** | Recency, Frequency, Monetary scoring with labelled segments |
| **Customer Lifetime Value** | AOV × Frequency × Lifespan per customer |
| **Cohort Retention** | Month-over-month retention heatmap by acquisition cohort |
| **Pareto Analysis** | 80/20 revenue concentration across the customer base |
| **Business Intelligence** | Revenue by country, segment, top products, cancellation analysis |
| **Streamlit Dashboard** | Six-page interactive dashboard for non-technical stakeholders |

---

## Architecture

```
Raw CSV
   │
   ▼
DataIngestor          — Load and validate file path
   │
   ▼
DataCleaner           — Schema validation, null handling, type standardisation
   │
   ▼
FeatureEngineering    — TotalRevenue, CancelledInvoice, YearMonth, CohortIndex
   │
   ▼
CustomerBehavior      — RFM, CLV, Cohort, Pareto
BusinessIntelligence  — Revenue by segment/country, top products, cancellations
   │
   ▼
DatasetBuilder        — Merges RFM + CLV + churn label into ML-ready dataset
   │
   ▼
ChurnModel            — Train, evaluate, cross-validate, predict, save
   │
   ▼
dashboard.py          — Streamlit dashboard (7 pages)
```

Each layer returns a dictionary of outputs. No layer mutates the output of another. A failure in one step raises immediately with a descriptive error — partial results are never passed downstream silently.

---

## Project Structure

```
analytics/
│
├── ingest.py                          # DataIngestor
├── clean.py                           # DataCleaner
├── features.py                        # FeatureEngineering
├── main.py                            # Entry point (analytics pipeline)
├── dashboard.py                       # Streamlit dashboard
├── sales.csv                          # Raw data
│
├── pipeline/
│   ├── analytics/
│   │   ├── customer_behavior.py       # RFM, CLV, Cohort, Pareto
│   │   ├── business_intelligence.py   # BI analytics
│   │   └── visualization_class.py     # Chart generation
│   │
│   └── ml/
│       ├── dataset_builder.py         # Builds ML dataset with churn labels
│       ├── model.py                   # ChurnModel class
│       └── run_churn_model.py         # ML pipeline entry point
│
└── reusables/
    └── reusable_functions.py          # Shared logger, schema validation
```

---

## Churn Model

### Features

| Feature | Description | Direction |
|---|---|---|
| Lifespan | Months since first purchase | Negative coefficient |
| RevenueTrend | Revenue change over last 90 days | Negative = at risk |
| AvgGapDays | Average days between purchases | Positive = at risk |
| Frequency | Total number of purchases | Negative coefficient |
| CLV | Customer lifetime value | Negative coefficient |

### Performance

| Metric | Score |
|---|---|
| Cross-validated F1 | 0.77 |
| Recall (churners) | 0.81 |
| Precision (churners) | 0.72 |
| Decision threshold | 0.4 |
| Validation | 5-fold stratified cross-validation |

### Churn Label Definition

A customer is labelled churned when their days since last purchase exceeds twice their personal average gap between purchases, with a minimum floor of 60 days. This approach respects each customer's established purchase pattern rather than applying a fixed threshold to all customers equally.

```python
ChurnThreshold = max(AvgGapDays × 2, 60)
Churned = Recency > ChurnThreshold
```

---

## Data Requirements

| Column | Type | Description |
|---|---|---|
| `CustomerNo` | string | Unique customer identifier |
| `TransactionNo` | string | Transaction identifier (prefix `C` = cancellation) |
| `Date` | date | Transaction date (format: MM/DD/YYYY) |
| `ProductNo` | string | Product identifier |
| `ProductName` | string | Product name |
| `Price` | float | Unit price |
| `Quantity` | int | Units purchased |
| `Country` | string | Customer country |

---

## Setup

```bash
# Clone the repository
git clone https://github.com/Isabirye-alex/analytics.git
cd analytics

# Install dependencies
pip install -r requirements.txt

# Run the analytics pipeline
python main.py

# Run the churn model
python pipeline/ml/run_churn_model.py

# Launch the dashboard
streamlit run dashboard.py
```

---

## Dashboard Pages

| Page | Audience Use Case |
|---|---|
| Overview | High-level KPIs — revenue, customers, churn rate |
| Churn Intelligence | At-risk customers, model confidence, priority retention list |
| Customer Segments | RFM segment sizes and revenue breakdown |
| Revenue Analysis | Pareto curve, country revenue, top products |
| Business Intelligence | Segment/country revenue, cancellations, top customers |
| Retention Heatmap | Cohort-based month-over-month retention |
| Customer Lifetime Value | CLV distribution, top customers with churn status |

---

## Design Principles

**Modularity** — Every class has a single responsibility. Swap a classifier, change a data source, or add an analytics step without touching unrelated code.

**Pipeline injection** — Classifiers and pipelines are injected as arguments rather than hardcoded inside classes. The `ChurnModel` accepts any sklearn-compatible Pipeline.

**Fail loudly** — Schema validation failures halt the pipeline immediately. Step-level failures in `BusinessIntelligence` are isolated and logged without stopping other steps.

**No leakage** — `Recency`, `ChurnThreshold`, and derived identifiers are excluded from the feature matrix via `DROP_COLUMNS`. The churn label is never visible to the model as a feature.

**Reproducibility** — `random_state=42` is set consistently. Stratified splits preserve class ratios. Cross-validation results are reported alongside single-split results.

---

## Known Limitations

- CLV model is simplified (AOV × Frequency × Lifespan). A probabilistic model such as BG/NBD + Gamma-Gamma would improve accuracy for customers with irregular purchase patterns.
- The pipeline currently reads from local CSV. PostgreSQL ingestion is designed and partially implemented via `save_to_postgres` in `DataIngestor`.
- The churn threshold is behavioural but static — it does not account for seasonality in purchase patterns.

---

## Author

Built as a full-stack data science project combining data engineering, analytics modelling, machine learning, and software architecture principles.