import logging
from typing import Callable, Dict, Any, List
import pandas as pd
from reusables.reusable_functions import ReusableFunctions


class DataCleaner:
    """
    DataCleaner is a production-grade utility for cleaning and validating
    transactional datasets.

    This class orchestrates a sequence of data preprocessing steps including:
        - Schema validation
        - Missing value analysis
        - Data type standardization
        - Invalid record removal
        - Memory optimization

    The pipeline also tracks data quality metrics such as:
        - Initial dataset size
        - Rows dropped per cleaning step
        - Null value distribution before and after cleaning

    Attributes:
        df (pd.DataFrame): Working copy of the dataset being cleaned.
        logger (logging.Logger): Logger instance for tracking pipeline execution.
        quality_metrics (Dict[str, Any]): Dictionary storing data quality insights.
    """

    REQUIRED_COLUMNS = [
        "Date",
        "ProductNo",
        "TransactionNo",
        "ProductName",
        "Price",
        "Quantity",
        "CustomerNo",
        "Country",
    ]

    def __init__(self, dataframe):
        """
        Initialize the DataCleaner with a dataset.

        A copy of the dataset is created to ensure that the original
        data remains unchanged during processing.

        Args:
            dataframe (pd.DataFrame): Raw input dataset to be cleaned.
        """

        # Create a defensive copy to prevent mutation of original dataset
        self.df = dataframe.copy()

        # Initialize reusable logger with class-specific name
        self.logger = ReusableFunctions.setup_logger(self.__class__.__name__)

        # Initialize data quality tracking structure
        self.quality_metrics: Dict[str, Any] = {
            "initial_row_count": len(self.df),  # Total rows before cleaning
            "rows_dropped": {},  # Tracks rows removed at each step
            "null_rates_before": {},  # Missing value ratios before cleaning
            "null_rates_after": {},  # Missing value ratios after cleaning
        }

        self.pipeline_steps: List[Callable] = [
            self._clean_dates,
            self._clean_customer_no,
            self._clean_country,
            self._clean_product_name,
        ]

    def _validate_schema(self) -> None:
        """
        Validate that all required columns exist in the dataset.

        Uses:
            ReusableFunctions.validate_schema

        Raises:
            ValueError:
                If any required column is missing.
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
            quality_metrics=self.quality_metrics,
            stage=stage,
            logger=self.logger,
        )

    def _clean_dates(self) -> None:
        """
        Standardize the 'Date' column.

        Converts the 'Date' column into pandas datetime format using a fixed format.
        Any values that cannot be parsed are coerced into NaT (Not a Time).

        Notes:
            - Invalid or malformed dates will not raise errors but will become NaT.
            - This ensures pipeline robustness while preserving row structure.
        """

        self.logger.info("Cleaning 'Date' column")

        self.df["Date"] = pd.to_datetime(
            self.df["Date"],
            format="%m/%d/%Y",
            errors="coerce",  # Invalid parsing results in NaT
        )

    def _clean_customer_no(self) -> None:
        """
        Clean and standardize the 'CustomerNo' column.

        Processing steps:
            1. Convert values to numeric format
            2. Remove rows with invalid or missing customer IDs
            3. Convert valid IDs to string format for consistency

        Data Quality Impact:
            - Ensures all customer identifiers are valid
            - Removes corrupted or missing identifiers
        """

        self.logger.info("Cleaning 'CustomerNo' column")

        before = len(self.df)

        # Convert to numeric; invalid values become NaN
        self.df["CustomerNo"] = pd.to_numeric(self.df["CustomerNo"], errors="coerce")

        # Drop rows with missing customer IDs
        self.df.dropna(subset=["CustomerNo"], inplace=True)

        dropped = before - len(self.df)

        # Record number of dropped rows for reporting
        self.quality_metrics["rows_dropped"]["invalid_customer_no"] = dropped

        # Convert to standardized string format
        self.df["CustomerNo"] = self.df["CustomerNo"].astype("Int64").astype(str)

        self.logger.info(f"Removed {dropped} rows due to invalid 'CustomerNo'")

    def _clean_country(self) -> None:
        """
        Standardize the 'Country' column.

        Processing steps:
            - Trim leading and trailing whitespace
            - Convert values to title case (e.g., 'uganda' → 'Uganda')

        Purpose:
            Ensures consistent formatting for grouping, filtering, and reporting.
        """

        self.logger.info("Cleaning 'Country' column")

        self.df["Country"] = self.df["Country"].str.strip().str.title()

    def _clean_product_name(self) -> None:
        """
        Validate and clean 'ProductName' entries.

        Removes rows where the product name contains only alphabetic characters,
        assuming such entries are incomplete or invalid.

        Example:
            - "Apple" → Removed (too generic / invalid per rule)
            - "Apple iPhone 13" → Kept

        Notes:
            - The validation rule can be adjusted based on business requirements.
        """

        self.logger.info("Cleaning 'ProductName' column")

        before = len(self.df)

        # Remove rows with overly simplistic product names
        self.df = self.df[~self.df["ProductName"].str.match(r"^[A-Za-z]+$", na=False)]

        dropped = before - len(self.df)

        self.quality_metrics["rows_dropped"]["invalid_product_name"] = dropped

        self.logger.info(f"Removed {dropped} rows due to invalid 'ProductName'")

    def _optimize_categories(self) -> None:
        """
        Optimize memory usage by converting selected columns to categorical dtype.

        Columns optimized:
            - ProductNo
            - Country
            - ProductName

        Benefits:
            - Reduces memory footprint
            - Improves performance for grouping and filtering operations
        """

        self.logger.info("Optimizing categorical columns")

        categorical_columns = ["ProductNo", "Country", "ProductName"]

        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete data cleaning pipeline.

        Pipeline Steps:
            1. Validate dataset schema
            2. Capture null value distribution (before cleaning)
            3. Perform column-level cleaning operations
            4. Optimize data types for efficiency
            5. Capture null value distribution (after cleaning)

        Returns:
            Dict[str, Any]:
                A dictionary containing:
                    - 'cleaned_dataset' (pd.DataFrame):
                        The processed and cleaned dataset.
                    - 'data_quality_metrics' (Dict):
                        Summary of data quality before and after cleaning.

        Raises:
            ValueError:
                If required columns are missing in the dataset.
        """

        self.logger.info("Starting data cleaning pipeline")

        # Step 1: Validate dataset schema
        self._validate_schema()
        # Step 2: Capture null distribution before cleaning
        self._capture_null_rates(stage='before')
        
        # Step 3: Execute cleaning steps
        for step in self.pipeline_steps:
            step()
        # Step 4: Optimize memory usage
        self._optimize_categories()

        # Step 5: Capture null distribution after cleaning
        self._capture_null_rates(stage='after')
        self.logger.info("Data cleaning pipeline completed successfully")

        return {
            "cleaned_dataset": self.df,
            "data_quality_metrics": self.quality_metrics,
        }
