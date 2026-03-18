# Customer Behavior Analytics Pipeline
# Overview

This project implements a modular, pipeline-driven customer analytics system for transactional datasets. It transforms raw sales data into actionable insights including:

RFM Segmentation (Recency, Frequency, Monetary)

Pareto Analysis (80/20 revenue rule)

Customer Lifetime Value (CLV)

Cohort Retention Analysis

The system is designed with scalability, reusability, and traceability in mind.

# Architecture

The pipeline follows a structured data flow:

Data Ingestion → Data Cleaning → Feature Engineering → Customer Behavior Analytics → Visualization
Core Components
Layer	Responsibility
DataInjestor	Loads raw data from CSV
DataCleaner	Handles missing values, formatting, and quality checks
FeatureEngineering	Creates derived features (cohorts, revenue, etc.)
CustomerBehavior	Computes analytics (RFM, CLV, Pareto, Cohort)
DataVisualization	Generates plots and insights

# Outputs
1. RFM Table

Customer segmentation based on:

Recency → Last purchase

Frequency → Number of transactions

Monetary → Total spend

2. Pareto Analysis

Identifies top revenue contributors

Validates 80/20 rule

3. Customer Lifetime Value (CLV)

# Formula:

CLV = AOV × Frequency × Lifespan
4. Cohort Retention Matrix

Tracks customer retention over time

Helps identify churn trends

# Data Requirements

The dataset must contain:

Column	Description
CustomerNo	Unique customer ID
TransactionNo	Transaction identifier
Date	Transaction date
TotalRevenue	Revenue per transaction
CancelledInvoice	Boolean flag
CohortMonth	First purchase month
CohortIndex	Months since first purchase
# Design Principles

Modularity → Each step is isolated and reusable

Pipeline Execution → Ordered step execution

Schema Validation → Prevents invalid inputs

Logging → Tracks execution flow

Extensibility → Easy to add new analytics steps

# Limitations

CLV model is simplified (not probabilistic)

No predictive modeling (yet)

Assumes clean transactional data

# Future Improvements

Churn prediction model (classification)

Advanced CLV models (BG/NBD, Gamma-Gamma)

Real-time data pipeline integration

API deployment for analytics serving

# Author

Built as part of a data science and software engineering pipeline project, combining:

Data Engineering principles

Analytics modeling

Clean architecture design