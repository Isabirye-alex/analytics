import pandas as pd
from clean import DataCleaner
from injest import DataInjestor
from features import FeatureEngineering
from pipeline.analytics.customer_behavior import CustomerBehavior
from pipeline.analytics.visualization_class import DataVisualization

file_path = "sales.csv"
# string = 'http://string.com'
dataframe = DataInjestor(file_path).fetch_data_from_csv()
cleaned_dataframe = DataCleaner(dataframe).run_pipeline()
feature_engineered_dataframe = FeatureEngineering(
    cleaned_dataframe["cleaned_dataset"]
).run_pipeline()
rfm_table = CustomerBehavior(feature_engineered_dataframe).build_rfm()
cohort_table = CustomerBehavior(feature_engineered_dataframe).build_cohort()
pareto = CustomerBehavior(feature_engineered_dataframe).build_pareto()
pareto_curve = DataVisualization().plot_pareto(pareto)
retention_heatmap = DataVisualization().plot_retention(cohort_table)
clv_table = CustomerBehavior(feature_engineered_dataframe).build_clv()
print(cleaned_dataframe["cleaned_dataset"].info())
print(
    "----------------------------------------------------------------------------------------------------------------------------------"
)
print(cleaned_dataframe["cleaned_dataset"].head())
print(
    "----------------------------------------------------------------------------------------------------------------------------------"
)
print(cleaned_dataframe["data_quality_metrics"])
print(
    "----------------------------------------------------------------------------------------------------------------------------------"
)
print(feature_engineered_dataframe.head())
print(cleaned_dataframe["cleaned_dataset"].info())
print(rfm_table)
print(
    "----------------------------------------------------------------------------------------------------------------------------------"
)
print(pareto)
print(
    ".............................................................................................."
)
print(clv_table.head(10))
print(
    "....................................................................................."
)
print(cohort_table)
