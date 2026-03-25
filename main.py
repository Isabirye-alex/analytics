import pandas as pd
from clean import DataCleaner
from ingest import DataIngestor
from features import FeatureEngineering
from pipeline.analytics.customer_behavior import CustomerBehavior
from pipeline.ml.dataset_builder import DatasetBuilder
from pipeline.ml.model import ChurnModel


def run_pipeline(file_path: str) -> dict:
    """
    Runs the full data pipeline (NO side effects).
    Returns all intermediate datasets.
    """

    # 1. Data Ingestion
    df = DataIngestor(file_path).fetch_data_from_csv()
    if isinstance(df, Exception):
        raise df

    # 2. Data Cleaning
    cleaning_output = DataCleaner(df).run_pipeline()
    cleaned_df = cleaning_output["cleaned_dataset"]

    # 3. Feature Engineering
    feature_df = FeatureEngineering(cleaned_df).run_pipeline()

    # 4. Customer Behavior Analytics
    cb_results = CustomerBehavior(feature_df).run_pipeline()

    rfm_table = cb_results["rfm"]
    pareto = cb_results["pareto"]
    cohort_table = cb_results["cohort"]
    clv_table = cb_results["clv"]
    # metrics = cb_results.get("tracking_metrics", {})

    # 5. Build ML Dataset
    dataset = DatasetBuilder(rfm_table, clv_table, feature_df).build()

    # 🔒 IMPORTANT: Make cohort serializable (for Streamlit caching)
    cohort_serialisable = cohort_table.copy()
    cohort_serialisable.index = cohort_serialisable.index.astype(str)
    cohort_serialisable.columns = cohort_serialisable.columns.astype(str)

    return {
        "raw": df.copy(),
        "cleaned": cleaned_df.copy(),
        "features": feature_df.copy(),
        "rfm": rfm_table.copy(),
        "pareto": pareto.copy(),
        "cohort": cohort_serialisable,
        "clv": clv_table.copy(),
        "dataset": dataset.copy(),
        # "metrics": metrics,
    }


def train_model(dataset: pd.DataFrame, threshold: float = 0.4) -> dict:
    """
    Trains churn model (NO file saving).
    """

    model = ChurnModel(dataset)

    # Train
    model.train(threshold=threshold)

    # Predict
    churn_scores = model.predict(threshold=threshold)

    # Feature importance
    feature_importance = model.get_feature_importance()

    return {
        "model": model,  # safe for st.cache_resource
        "churn_scores": churn_scores,
        "feature_importance": feature_importance,
    }


# Optional CLI usage (safe)
if __name__ == "__main__":
    results = run_pipeline("sales.csv")
    model_results = train_model(results["dataset"])

