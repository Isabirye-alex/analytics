# Customer Analytics & Churn Intelligence Pipeline

A production-ready analytics system that turns raw retail transactions into actionable insights: data cleaning, customer segmentation, churn prediction, and an interactive dashboard.

# Cell 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

## 1. Load Dataset

Download the dataset from Kaggle: [Online Retail Dataset](https://www.kaggle.com/code/ekajaya/analysis-dataset-sales-transaction-v-4a-csv)

# Cell 2: Load CSV
df = pd.read_csv("sales.csv", parse_dates=['Date'])
df.head()

## 2. Data Overview

- 536,350 transactions  
- 4,738 unique customers  
- Mostly UK customers, some from 38 other countries  
- Columns: TransactionNo, Date, ProductNo, ProductName, Price, Quantity, CustomerNo, Country

# Quick stats
df.info()
df.describe()
df['Country'].value_counts().head(10)

## 3. Data Cleaning

- Remove invalid `CustomerNo` and `ProductName` rows  
- Handle cancellations (`Quantity < 0`)  
- Remove duplicates and missing values

# Remove invalid CustomerNo and ProductName
df_clean = df.dropna(subset=['CustomerNo', 'ProductName'])

# Separate cancellations
df_clean['Cancelled'] = df_clean['Quantity'] < 0

# Optional: remove cancellations for certain analyses
df_no_cancel = df_clean[~df_clean['Cancelled']]

df_clean.shape, df_no_cancel.shape

## 4. Feature Engineering

- TotalRevenue = Price × Quantity  
- Recency, Frequency, Monetary (RFM) scores  
- Cohort month, CLV approximation

# Total Revenue
df_clean['TotalRevenue'] = df_clean['Price'] * df_clean['Quantity']

# Example: RFM features
rfm = df_clean.groupby('CustomerNo').agg({
    'Date': lambda x: (df_clean['Date'].max() - x.max()).days,  # Recency
    'TransactionNo': 'count',  # Frequency
    'TotalRevenue': 'sum'  # Monetary
}).rename(columns={'Date':'Recency','TransactionNo':'Frequency','TotalRevenue':'Monetary'})

rfm.head()

## 5. Define Churn

- Churn if days since last purchase > max(2 × AvgGapDays, 60)

# Calculate average gap days per customer
df_clean = df_clean.sort_values(['CustomerNo', 'Date'])
df_clean['PrevDate'] = df_clean.groupby('CustomerNo')['Date'].shift(1)
df_clean['GapDays'] = (df_clean['Date'] - df_clean['PrevDate']).dt.days
avg_gap = df_clean.groupby('CustomerNo')['GapDays'].mean().fillna(0)
last_purchase = df_clean.groupby('CustomerNo')['Date'].max()

churn_threshold = np.maximum(avg_gap*2, 60)
churned = (pd.to_datetime('2011-12-31') - last_purchase).dt.days > churn_threshold
rfm['Churned'] = churned.astype(int)
rfm.head()

## 6. Model Training

- Gradient Boosting classifier  
- Target: `Churned`  
- Features: Recency, Frequency, Monetary, CLV, RevenueTrend, AvgGapDays

# Example model
features = ['Recency','Frequency','Monetary']  # extend with other engineered features
X = rfm[features]
y = rfm['Churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("F1:", f1_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))

## 7. Visualizations

- RFM distribution  
- Pareto chart (top 20% customers by revenue)  
- Cohort retention heatmap

# Example: Pareto chart
top_customers = rfm.sort_values('Monetary', ascending=False)
top20 = top_customers['Monetary'].cumsum() / top_customers['Monetary'].sum()
plt.figure(figsize=(10,5))
plt.plot(top20)
plt.axhline(0.8, color='red', linestyle='--')
plt.title("Pareto Analysis: Cumulative Revenue")
plt.xlabel("Customers sorted by revenue")
plt.ylabel("Cumulative revenue proportion")
plt.show()

## 8. Dashboard

Use Streamlit for interactive visualization:

```bash
streamlit run dashboard.py

## Author

Full-stack data science project combining **data engineering, analytics, machine learning, and interactive dashboards**.