import pandas as pd
from reusables.reusable_functions import ReusableFunctions
from typing import List, Dict, Callable, Any


class BusinessIntelligenceError(Exception):
    """Base exception for BusinessIntelligence pipeline failures."""

    pass


class SchemaValidationError(BusinessIntelligenceError):
    """Raised when required columns are missing from the dataset."""

    pass


class PipelineStepError(BusinessIntelligenceError):
    """Raised when an individual pipeline step fails."""

    pass


class BusinessIntelligence:
    """
    BusinessIntelligence generates operational analytics from transactional
    and RFM data.

    Computes:
        - Revenue by country
        - Revenue by customer segment
        - Top selling products
        - Top customers by revenue
        - Top products per country
        - Top products per segment

    Design Principles:
        - Modular pipeline execution
        - Separation of concerns
        - Reusability via utility functions
        - Traceability through logging and metrics
    """

    REQUIRED_COLUMNS = ["CustomerNo", "Country", "ProductName", "TotalRevenue"]
    REQUIRED_RFM_COLUMNS = ["CustomerNo", "Segment", "Monetary"]

    def __init__(self, dataframe: pd.DataFrame, rfm: pd.DataFrame) -> None:
        """
        Initialize BusinessIntelligence pipeline.

        Args:
            dataframe (pd.DataFrame): Feature-engineered transactional dataset.
            rfm (pd.DataFrame): RFM table with Segment and Monetary columns.

        Raises:
            TypeError: If inputs are not DataFrames.
            ValueError: If either DataFrame is empty.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pd.DataFrame for dataframe, got {type(dataframe).__name__}"
            )

        if not isinstance(rfm, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame for rfm, got {type(rfm).__name__}")

        if dataframe.empty:
            raise ValueError("dataframe is empty — cannot run pipeline.")

        if rfm.empty:
            raise ValueError("rfm table is empty — cannot run pipeline.")

        self.df = dataframe.copy()
        self.rfm_table = rfm.copy()
        self.logger = ReusableFunctions.setup_logger(self.__class__.__name__)

        # Output storage — populated by pipeline steps
        self.revenue_by_country: pd.DataFrame | None = None
        self.revenue_by_segment: pd.DataFrame | None = None
        self.top_products: pd.DataFrame | None = None
        self.top_products_by_country: pd.DataFrame | None = None
        self.top_products_by_segment: pd.DataFrame | None = None
        self.top_customers: pd.DataFrame | None = None
        self.top_cancelling_segments: pd.DataFrame | None = None
        self.top_cancelling_countries: pd.DataFrame | None = None

        # Add to REQUIRED_COLUMNS
        REQUIRED_COLUMNS = [
            "CustomerNo", "Country", "ProductName",
            "TotalRevenue", "CancelledInvoice"  # Add this
        ]

        self.metrics: Dict[str, Any] = {
            "steps_completed": [],
            "steps_failed": [],
        }

        self.pipeline_steps: List[Callable] = [
            self._revenue_by_country,
            self._revenue_by_segment,
            self._top_customers,
            self._top_products,
            self._top_products_by_country,
            self._top_products_by_segment,
            self._top_cancelling_segments,  
            self._top_cancelling_countries,
        ]

    # Schema Validation

    def _validate_schemas(self) -> None:
        """
        Validate that all required columns exist in both DataFrames
        before any pipeline step executes.

        Raises:
            SchemaValidationError: If any required column is missing.
        """

        missing_main = [
            col for col in self.REQUIRED_COLUMNS if col not in self.df.columns
        ]

        if missing_main:
            raise SchemaValidationError(
                f"Main dataframe missing required columns: {missing_main}"
            )

        missing_rfm = [
            col
            for col in self.REQUIRED_RFM_COLUMNS
            if col not in self.rfm_table.columns
        ]

        if missing_rfm:
            raise SchemaValidationError(
                f"RFM table missing required columns: {missing_rfm}"
            )

        self.logger.info("Schema validation passed")

    # Pipeline Steps

    def _revenue_by_country(self) -> pd.DataFrame:
        """
        Total revenue generated per country, descending.

        Raises:
            PipelineStepError: If aggregation fails.
        """

        self.logger.info("Computing revenue by country")

        try:
            self.revenue_by_country = (
                self.df.groupby("Country", observed=False)["TotalRevenue"]
                .sum()
                .reset_index()
                .sort_values("TotalRevenue", ascending=False)
                .reset_index(drop=True)
            )
            return self.revenue_by_country

        except Exception as e:
            raise PipelineStepError(f"_revenue_by_country failed: {e}") from e

    def _revenue_by_segment(self) -> pd.DataFrame:
        """
        Total revenue per RFM customer segment, descending.

        Raises:
            PipelineStepError: If aggregation fails.
        """

        self.logger.info("Computing revenue by segment")

        try:
            self.revenue_by_segment = (
                self.rfm_table.groupby("Segment", observed=False)["Monetary"]
                .sum()
                .reset_index()
                .sort_values("Monetary", ascending=False)
                .reset_index(drop=True)
            )
            return self.revenue_by_segment

        except Exception as e:
            raise PipelineStepError(f"_revenue_by_segment failed: {e}") from e

    def _top_customers(self) -> pd.DataFrame:
        """
        Top customers ranked by total revenue generated.

        Raises:
            PipelineStepError: If aggregation fails.
        """

        self.logger.info("Computing top customers by revenue")

        try:
            self.top_customers = (
                self.df.groupby("CustomerNo")["TotalRevenue"]
                .sum()
                .reset_index()
                .sort_values("TotalRevenue", ascending=False)
                .reset_index(drop=True)
            )
            self.top_customers = self.top_customers.head(50)
            return self.top_customers

        except Exception as e:
            raise PipelineStepError(f"_top_customers failed: {e}") from e

    def _top_products(self) -> pd.DataFrame:
        """
        Top products ranked by total revenue generated.

        Raises:
            PipelineStepError: If aggregation fails.
        """

        self.logger.info("Computing top products by revenue")

        try:
            self.top_products = (
                self.df.groupby("ProductName", observed=False)["TotalRevenue"]
                .sum()
                .reset_index()
                .sort_values("TotalRevenue", ascending=False)
                .reset_index(drop=True)
            )
            self.top_products = self.top_products.head(50)
            return self.top_products

        except Exception as e:
            raise PipelineStepError(f"_top_products failed: {e}") from e

    def _top_products_by_country(self) -> pd.DataFrame:
        """
        Top revenue-generating product per country.

        For each country, identifies the single product that generated
        the most revenue.

        Raises:
            PipelineStepError: If aggregation or grouping fails.
        """

        self.logger.info("Computing top product per country")

        try:
            product_country = (
                self.df.groupby(["Country", "ProductName"], observed=False)[
                    "TotalRevenue"
                ]
                .sum()
                .reset_index()
            )

            self.top_products_by_country = (
                product_country.sort_values("TotalRevenue", ascending=False)
                .groupby("Country", observed=False)
                .first()
                .reset_index()
                .sort_values("TotalRevenue", ascending=False)
                .reset_index(drop=True)
            )
            return self.top_products_by_country

        except Exception as e:
            raise PipelineStepError(f"_top_products_by_country failed: {e}") from e

    def _top_products_by_segment(self) -> pd.DataFrame:
        """
        Top revenue-generating product per customer segment.

        Merges transactional data with RFM segment labels, then
        identifies the highest revenue product within each segment.

        Raises:
            PipelineStepError: If merge or aggregation fails.
        """

        self.logger.info("Computing top product per segment")

        try:
            df_with_segment = self.df.merge(
                self.rfm_table[["CustomerNo", "Segment"]],
                on="CustomerNo",
                how="left",
            )

            # Warn if any customers failed to match a segment
            unmatched = df_with_segment["Segment"].isna().sum()
            if unmatched > 0:
                self.logger.warning(
                    f"{unmatched} transactions could not be matched to a segment"
                )

            product_segment = (
                df_with_segment.groupby(["Segment", "ProductName"], observed=False)[
                    "TotalRevenue"
                ]
                .sum()
                .reset_index()
            )

            self.top_products_by_segment = (
                product_segment.sort_values("TotalRevenue", ascending=False)
                .groupby("Segment", observed=False)
                .first()
                .reset_index()
                .sort_values("TotalRevenue", ascending=False)
                .reset_index(drop=True)
            )
            return self.top_products_by_segment

        except Exception as e:
            raise PipelineStepError(f"_top_products_by_segment failed: {e}") from e

    def _top_cancelling_segments(self) -> pd.DataFrame:
        """
        Identify which customer segments cancel the most invoices.

        Merges cancellation flags from the transactional data with
        RFM segment labels, then ranks segments by cancellation count.

        Raises:
            PipelineStepError: If merge or aggregation fails.
        """

        self.logger.info("Computing cancellation rate by segment")

        try:
            cancelled = self.df[self.df["CancelledInvoice"] == True].copy()

            if cancelled.empty:
                self.logger.warning("No cancelled invoices found in dataset")
                self.top_cancelling_segments = pd.DataFrame()
                return self.top_cancelling_segments

            df_with_segment = cancelled.merge(
                self.rfm_table[["CustomerNo", "Segment"]],
                on="CustomerNo",
                how="left",
            )

            unmatched = df_with_segment["Segment"].isna().sum()
            if unmatched > 0:
                self.logger.warning(
                    f"{unmatched} cancelled transactions could not be matched to a segment"
                )

            self.top_cancelling_segments = (
                df_with_segment.groupby("Segment", observed=False)
                .agg(
                    CancellationCount=("CancelledInvoice", "count"),
                    RevenueImpact=("TotalRevenue", "sum"),
                )
                .reset_index()
                .sort_values("CancellationCount", ascending=False)
                .reset_index(drop=True)
            )

            return self.top_cancelling_segments

        except Exception as e:
            raise PipelineStepError(
                f"_top_cancelling_segments failed: {e}"
            ) from e

    def _top_cancelling_countries(self) -> pd.DataFrame:
        """
        Identify which countries cancel the most invoices.

        Filters to cancelled transactions only, then ranks countries
        by cancellation count and revenue impact.

        Raises:
            PipelineStepError: If aggregation fails.
        """

        self.logger.info("Computing cancellation rate by country")

        try:
            cancelled = self.df[self.df["CancelledInvoice"] == True].copy()

            if cancelled.empty:
                self.logger.warning("No cancelled invoices found in dataset")
                self.top_cancelling_countries = pd.DataFrame()
                return self.top_cancelling_countries

            self.top_cancelling_countries = (
                cancelled.groupby("Country", observed=False)
                .agg(
                    CancellationCount=("CancelledInvoice", "count"),
                    RevenueImpact=("TotalRevenue", "sum"),
                )
                .reset_index()
                .sort_values("CancellationCount", ascending=False)
                .reset_index(drop=True)
            )

            return self.top_cancelling_countries

        except Exception as e:
            raise PipelineStepError(
                f"_top_cancelling_countries failed: {e}"
            ) from e
        # Orchestrator

    def run_pipeline(self) -> Dict[str, pd.DataFrame]:
        """
        Execute all analytics steps in sequence.

        Validates schemas first. Each step is executed independently —
        a failure in one step is logged and recorded in metrics without
        halting the remaining steps. Results for failed steps will be None.

        Returns:
            Dict[str, pd.DataFrame]: Named results from each step.
                Failed steps return None for their key.

        Raises:
            SchemaValidationError: If required columns are missing.
                This halts the entire pipeline before any steps run.
        """

        self.logger.info("Starting BusinessIntelligence pipeline")

        # Schema failure halts everything — no point proceeding
        self._validate_schemas()

        for step in self.pipeline_steps:
            step_name = step.__name__
            try:
                step()
                self.metrics["steps_completed"].append(step_name)
                self.logger.info(f"Step completed: {step_name}")

            except PipelineStepError as e:
                self.metrics["steps_failed"].append(step_name)
                self.logger.error(f"Step failed: {step_name} — {e}")
                # Continue to next step rather than halting the pipeline
                continue

        completed = len(self.metrics["steps_completed"])
        failed = len(self.metrics["steps_failed"])

        self.logger.info(
            f"Pipeline finished — {completed} steps completed, "
            f"{failed} steps failed"
        )

        if failed > 0:
            self.logger.warning(f"Failed steps: {self.metrics['steps_failed']}")

        return {
            "revenue_by_country": self.revenue_by_country,
            "revenue_by_segment": self.revenue_by_segment,
            "top_customers": self.top_customers,
            "top_products": self.top_products,
            "top_products_by_country": self.top_products_by_country,
            "top_products_by_segment": self.top_products_by_segment,
            "top_cancelling_segments": self.top_cancelling_segments,  # Add
            "top_cancelling_countries": self.top_cancelling_countries,
            "metrics": self.metrics,
        }  # type: ignore
