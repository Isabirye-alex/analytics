import streamlit as st
import pandas as pd
from clean import DataCleaner
from features import FeatureEngineering
from ingest import DataIngestor
from pipeline.analytics.customer_behavior import CustomerBehavior
from pipeline.ml.dataset_builder import DatasetBuilder
from pipeline.ml.model import ChurnModel

# ... (keep your other imports)


def main():
    st.title("📊 Customer Analytics & Churn Prediction")

    file_path = "sales.csv"

    # Use a spinner so the user knows the app is working
    with st.spinner("Processing data and training model..."):
        # 1. Data Ingestion
        df = DataIngestor(file_path).fetch_data_from_csv()

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
        cohort = cb_results["cohort"]
        clv_table = cb_results["clv"]
        metrics = cb_results["tracking_metrics"]

        # Build and Train Model
        dataset = DatasetBuilder(rfm_table, clv_table, feature_df).build()
        model = ChurnModel(dataset)
        model.train(threshold=0.4)
        churn_scores = model.predict(threshold=0.4)

    # 5. DISPLAY RESULTS IN THE UI
    st.success("Analysis Complete!")

    # Create Tabs for a clean UI
    tab1, tab2, tab3 = st.tabs(
        ["Customer Metrics", "Visualizations", "Churn Predictions"]
    )

    with tab1:
        st.header("Key Metrics")
        st.dataframe(rfm_table.head())
        st.write(f"Total Customers Scored: {len(churn_scores)}")

    with tab2:
        st.header("Behavioral Analysis")
        # Display your plots
        st.pyplot(pareto)
        st.pyplot(cohort)

    with tab3:
        st.header("High Risk Customers")
        st.dataframe(churn_scores)


if __name__ == "__main__":
    main()
