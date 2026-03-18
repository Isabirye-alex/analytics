import pandas as pd
import logging
from typing import Dict, Any, List, Callable
from reusables.reusable_functions import ReusableFunctions


class FeatureEngineering:
    """
    FeatureEngineering is responsible for transforming a cleaned transactional dataset
    into a feature-rich dataset suitable for analytics and modeling.

    This includes:
        - Revenue calculations
        - Cancellation detection
        - Time-based aggregations
        - Customer cohort analysis

    The class also tracks feature-level data quality metrics.
    """

    REQUIRED_COLUMNS = ["TransactionNo", "Date", "Price", "Quantity", "CustomerNo"]

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the FeatureEngineering pipeline.

        Args:
            dataframe (pd.DataFrame): Cleaned dataset ready for feature generation.
        """

        self.df = dataframe.copy()

        # Use reusable logger
        self.logger = ReusableFunctions.setup_logger(self.__class__.__name__)

        # Feature quality metrics
        self.feature_metrics: Dict[str, Any] = {
            "initial_row_count": len(self.df),
            "rows_affected": {},
            "null_rates_before": {},
            "null_rates_after": {},
        }

        self.pipeline_steps :List[Callable] = [
            self._add_revenue_feature, 
            self._add_cancellation_flag,
            self._add_time_features,
            self._add_cohort_index
        ]

    def _validate_schema(self) -> None:
        """
        Validate required columns exist in the dataset.
        """
        ReusableFunctions.validate_schema(
            df=self.df,
            required_columns=self.REQUIRED_COLUMNS,
            logger=self.logger,
        )

    def _capture_null_rates(self, stage: str) -> None:
        """
        Capture null value distribution across all columns.

        Args:
            stage (str): 'before' or 'after'
        """
        ReusableFunctions.capture_null_rates(
            df=self.df,
            quality_metrics=self.feature_metrics,
            stage=stage,
            logger=self.logger,
        )

    def _add_revenue_feature(self) -> None:
        """
        Create TotalRevenue feature.

        Formula:
            TotalRevenue = Quantity * Price
        """
        self.logger.info("Adding TotalRevenue feature")
        self.df["TotalRevenue"] = self.df["Quantity"] * self.df["Price"]

    def _add_cancellation_flag(self) -> None:
        """
        Identify cancelled transactions.

        Logic:
            Transaction numbers starting with 'C' indicate cancellations.
        """
        self.logger.info("Adding CancelledInvoice feature")

        self.df["CancelledInvoice"] = (
            self.df["TransactionNo"].astype(str).str.startswith("C")
        )

        cancelled_count = self.df["CancelledInvoice"].sum()

        self.feature_metrics["rows_affected"]["cancelled_invoices"] = int(
            cancelled_count
        )

        self.logger.info(f"Total cancelled invoices: {cancelled_count}")

    def _add_time_features(self) -> None:
        """
        Generate time-based features from the Date column.

        Features:
            - YearMonth
            - CohortMonth
        """
        self.logger.info("Adding time-based features")

        self.df["YearMonth"] = self.df["Date"].dt.to_period("M")

        self.df["CohortMonth"] = self.df.groupby("CustomerNo", observed=False)[
            "YearMonth"
        ].transform("min")

    def _add_cohort_index(self) -> None:
        """
        Compute CohortIndex for cohort analysis.

        CohortIndex = months since first purchase.
        """
        self.logger.info("Calculating CohortIndex")

        invoice_year = self.df["YearMonth"].dt.year
        invoice_month = self.df["YearMonth"].dt.month
        cohort_year = self.df["CohortMonth"].dt.year
        cohort_month = self.df["CohortMonth"].dt.month

        self.df["CohortIndex"] = (
            (invoice_year - cohort_year) * 12 + (invoice_month - cohort_month) + 1
        )

    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the full feature engineering pipeline.

        Steps:
            1. Validate schema
            2. Capture null rates (before)
            3. Generate features
            4. Capture null rates (after)

        Returns:
            pd.DataFrame: Feature-engineered dataset
        """

        self.logger.info("Starting feature engineering pipeline")

        # 1. Schema validation (reusable)
        self._validate_schema()

        # 2. Null tracking (reusable)
        self._capture_null_rates("before")

        # 3. Feature generation
        for step in self.pipeline_steps:
            step()

        # 4. Null tracking after
        self._capture_null_rates("after")

        self.logger.info("Feature engineering pipeline completed successfully")

        return self.df
