import pandas as pd
import logging
from typing import Dict, Any
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

    def __init__(self, dataframe):
        """
        Initialize the FeatureEngineering pipeline.

        Args:
            dataframe (pd.DataFrame): Cleaned dataset ready for feature generation.
        """

        # Work on a copy to preserve original dataset integrity
        self.df = dataframe.copy()

        # Use reusable logger
        self.logger = ReusableFunctions.setup_logger(self.__class__.__name__)

        # Initialize feature-level quality metrics
        self.feature_metrics: Dict[str, Any] = {
            "initial_row_count": len(self.df),
            "rows_affected": {},  # Tracks rows impacted by transformations
            "null_rates_before": {},
            "null_rates_after": {},
        }

    def validate_schema(self) -> None:
        """
        Validate that all required columns exist before feature engineering.

        Raises:
            ValueError: If any required column is missing.
        """

        missing_columns = [
            col for col in self.REQUIRED_COLUMNS if col not in self.df.columns
        ]

        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError("Required columns missing")

        self.logger.info("Schema validation passed")

    def capture_null_rates(self, stage: str) -> None:
        """
        Capture null value distribution across all columns.

        Args:
            stage (str): 'before' or 'after' feature engineering.
        """

        null_rates = self.df.isna().mean().round(4).to_dict()

        if stage == "before":
            self.feature_metrics["null_rates_before"] = null_rates
        elif stage == "after":
            self.feature_metrics["null_rates_after"] = null_rates
        else:
            raise ValueError("Stage must be 'before' or 'after'")

        self.logger.info(f"Captured null rates ({stage})")

    def _add_revenue_feature(self) -> None:
        """
        Create TotalRevenue feature.

        Formula:
            TotalRevenue = Quantity * Price

        Purpose:
            Enables revenue-based analytics such as:
                - Customer value
                - Sales trends
                - Product performance
        """

        self.logger.info("Adding TotalRevenue feature")

        self.df["TotalRevenue"] = self.df["Quantity"] * self.df["Price"]

    def _add_cancellation_flag(self) -> None:
        """
        Identify cancelled transactions.

        Logic:
            Transaction numbers starting with 'C' indicate cancellations.

        Output:
            Adds a boolean column 'CancelledInvoice'

        Also logs total number of cancelled transactions.
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

        Features created:
            - YearMonth: Period representing year and month of transaction
            - CohortMonth: First purchase month for each customer

        Purpose:
            Enables time-series analysis and cohort tracking.
        """

        self.logger.info("Adding time-based features")

        # Convert to monthly period
        self.df["YearMonth"] = self.df["Date"].dt.to_period("M")

        # First purchase month per customer (cohort)
        self.df["CohortMonth"] = self.df.groupby("CustomerNo", observed=False)[
            "YearMonth"
        ].transform("min")

    def _add_cohort_index(self) -> None:
        """
        Compute CohortIndex for cohort analysis.

        CohortIndex represents the number of months since a customer's
        first transaction.

        Formula:
            CohortIndex = (Year difference * 12) + Month difference + 1

        Example:
            First purchase: Jan 2023
            Current purchase: Mar 2023
            CohortIndex = 3 months
        """

        self.logger.info("Calculating CohortIndex")

        invoice_year = self.df["YearMonth"].dt.year
        invoice_month = self.df["YearMonth"].dt.month
        cohort_year = self.df["CohortMonth"].dt.year
        cohort_month = self.df["CohortMonth"].dt.month

        years_diff = invoice_year - cohort_year
        months_diff = invoice_month - cohort_month

        self.df["CohortIndex"] = (years_diff * 12) + months_diff + 1

    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the full feature engineering pipeline.

        Steps:
            1. Validate schema
            2. Capture null rates (before)
            3. Generate revenue feature
            4. Detect cancelled invoices
            5. Create time-based features
            6. Compute cohort index
            7. Capture null rates (after)

        Returns:
            pd.DataFrame:
                Dataset enriched with engineered features.

        Raises:
            ValueError:
                If required columns are missing.
        """

        self.logger.info("Starting feature engineering pipeline")

        # Step 1: Validate schema
        self.validate_schema()

        # Step 2: Capture null rates before transformations
        self.capture_null_rates("before")

        # Step 3–6: Feature creation
        self._add_revenue_feature()
        self._add_cancellation_flag()
        self._add_time_features()
        self._add_cohort_index()

        # Step 7: Capture null rates after transformations
        self.capture_null_rates("after")

        self.logger.info("Feature engineering pipeline completed successfully")

        return self.df
