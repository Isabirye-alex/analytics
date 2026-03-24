import pandas as pd
from clean import DataCleaner
from injest import DataInjestor
from features import FeatureEngineering
from pipeline.analytics.customer_behavior import CustomerBehavior
from pipeline.analytics.visualization_class import DataVisualization
from pipeline.ml.dataset_builder import DatasetBuilder
from pipeline.ml.feature_selector import FeatureSelector
from pipeline.ml.model import ChurnModel


def main():
    file_path = "sales.csv"

     
    # 1. Data Ingestion
     
    df = DataInjestor(file_path).fetch_data_from_csv()

     
    # 2. Data Cleaning
     
    cleaning_output = DataCleaner(df).run_pipeline()
    cleaned_df = cleaning_output["cleaned_dataset"]

     
    # 3. Feature Engineering
    
    fe_pipeline = FeatureEngineering(cleaned_df)
    feature_df = fe_pipeline.run_pipeline()

    # 4. Customer Behavior 
    
    cb_pipeline = CustomerBehavior(feature_df)
    cb_results = cb_pipeline.run_pipeline()

    rfm_table = cb_results["rfm"]
    pareto = cb_results["pareto"]
    cohort_table = cb_results["cohort"]
    clv_table = cb_results["clv"]
    metrics = cb_results['tracking_metrics']

     
    # 5. Visualization
     
    viz = DataVisualization()

    pareto_curve = viz.plot_pareto(pareto)
    retention_heatmap = viz.plot_retention(cohort_table)


   # Build dataset
    dataset = DatasetBuilder(rfm_table, clv_table, feature_df).build()
    print(dataset.columns)

    # Train model
    model = ChurnModel(dataset).train(threshold=0.4)
     
    # 9. Debug / Inspection
     
    # print("\n=== CLEANED DATA INFO ===")
    # print(cleaned_df.info())

    # print("\n=== FEATURE DATA SAMPLE ===")
    # print(feature_df.head())

    # print("\n=== DATA QUALITY METRICS ===")
    # print(cleaning_output["data_quality_metrics"])

    # print("\n=== RFM TABLE ===")
    # print(rfm_table)

    # print("\n=== PARETO ===")
    # print(pareto.head())

    # print("\n=== CLV ===")
    # print(clv_table.head())

    # print("\n=== COHORT ===")
    # print(cohort_table)

    # print("\n=== Model Dataset ===")
    # print(dataset.head())


if __name__ == "__main__":
    main()
