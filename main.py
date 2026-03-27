import pandas as pd
from clean import DataCleaner
from ingest import DataIngestor
from features import FeatureEngineering
from pipeline.analytics.customer_behavior import CustomerBehavior
from pipeline.ml.dataset_builder import DatasetBuilder
from pipeline.ml.model import ChurnModel


def run_pipeline(file_path: str) -> dict:
    try:
        # 1. Data Ingestion
        try:
            df = DataIngestor().fetch_data_from_csv(file_path)
        except Exception as e:
            raise RuntimeError(f"[INGESTION ERROR] Failed to load file: {e}")

        # 2. Data Cleaning
        try:
            cleaning_output = DataCleaner(df).run_pipeline()
            cleaned_df = cleaning_output["cleaned_dataset"]
        except Exception as e:
            raise RuntimeError(f"[CLEANING ERROR] {e}")

        # 3. Feature Engineering
        try:
            feature_df = FeatureEngineering(cleaned_df).run_pipeline()
        except Exception as e:
            raise RuntimeError(f"[FEATURE ENGINEERING ERROR] {e}")

        # 4. Customer Behavior
        try:
            cb_results = CustomerBehavior(feature_df).run_pipeline()
        except Exception as e:
            raise RuntimeError(f"[CUSTOMER BEHAVIOR ERROR] {e}")

        # Extract safely
        try:
            rfm_table = cb_results["rfm"]
            pareto = cb_results["pareto"]
            cohort_table = cb_results["cohort"]
            clv_table = cb_results["clv"]
            metrics = cb_results.get("tracking_metrics", {})
        except KeyError as e:
            raise RuntimeError(f"[MISSING OUTPUT KEY] {e}")

        # 5. Dataset Builder
        try:
            dataset = DatasetBuilder(rfm_table, clv_table, feature_df).build()
        except Exception as e:
            raise RuntimeError(f"[DATASET BUILD ERROR] {e}")

        # Fix serialization
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
            "metrics": metrics,
        }

    except Exception as e:
        print(f"[PIPELINE FAILED] {e}")
        raise


def train_model(dataset: pd.DataFrame, threshold: float = 0.4) -> dict:
    try:
        model = ChurnModel(dataset)

        try:
            model.train(threshold=threshold)
        except Exception as e:
            raise RuntimeError(f"[MODEL TRAINING ERROR] {e}")

        try:
            churn_scores = model.predict(threshold=threshold)
        except Exception as e:
            raise RuntimeError(f"[PREDICTION ERROR] {e}")

        try:
            feature_importance = model.get_feature_importance()
        except Exception as e:
            raise RuntimeError(f"[FEATURE IMPORTANCE ERROR] {e}")

        return {
            "model": model,
            "churn_scores": churn_scores,
            "feature_importance": feature_importance,
        }

    except Exception as e:
        print(f"[MODEL PIPELINE FAILED] {e}")
        raise


def save_to_db(results: dict, connection_string: str):
    if not connection_string:
        print("[DB] No connection string provided. Skipping save.")
        return

    db_ingestor = DataIngestor()

    for table_name, df in results.items():
        if not isinstance(df, pd.DataFrame):
            continue

        try:
            db_ingestor.save_to_postgress(df, table_name, connection_string)
            print(f"[DB] Saved table: {table_name}")

        except Exception as e:
            print(f"[DB ERROR] Failed to save {table_name}: {e}")


if __name__ == "__main__":
    try:
        results = run_pipeline("sales.csv")

        model_results = train_model(results["dataset"])

        connection_string = "postgresql://postgres:0009@localhost:5432/ds_db"

        save_to_db(results, connection_string)

        print("Pipeline completed successfully")

    except Exception as e:
        print(f"SYSTEM FAILURE: {e}")
