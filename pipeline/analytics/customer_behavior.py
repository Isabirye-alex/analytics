import pandas as pd
from typing import Dict, Any, Callable, List
from reusables.reusable_functions import ReusableFunctions


class CustomerBehavior:
    """
    CustomerBehavior is a pipeline-driven analytics class that generates
    customer-level insights from transactional data.

    It computes:
        - RFM segmentation (Recency, Frequency, Monetary)
        - Pareto revenue distribution (80/20 rule)
        - Customer Lifetime Value (CLV)
        - Cohort retention analysis

    Design Principles:
        - Modular pipeline execution
        - Separation of concerns
        - Reusability via utility functions
        - Traceability through logging and metrics tracking
    """

    REQUIRED_COLUMNS = [
        "TotalRevenue",
        "CohortIndex",
        "CohortMonth",
        "CustomerNo",
        "Date",
        "TransactionNo",
        "CancelledInvoice",
    ]

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the CustomerBehavior pipeline.

        Args:
            dataframe (pd.DataFrame):
                Feature-engineered dataset ready for analytics.
        """

        # Work on a copy to avoid mutating upstream data
        self.df = dataframe.copy()

        # Initialize reusable logger
        self.logger = ReusableFunctions.setup_logger(self.__class__.__name__)

        # Output containers (populated during pipeline execution)
        self.rfm_table: pd.DataFrame | None = None
        self.pareto: pd.DataFrame | None = None
        self.clv_table: pd.DataFrame | None = None
        self.cohort: pd.DataFrame | None = None

        # Execution tracking
        self.metrics: Dict[str, Any] = {
            "initial_row_count": len(self.df),
            "steps_executed": [],
        }

        # Ordered pipeline steps (execution order matters)
        self.pipeline_steps: List[Callable] = [
            self._build_rfm,
            self._build_pareto,
            self._build_clv,
            self._build_cohort,
        ]


    # Validation Layer (Reusable)


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


    # Pipeline Steps


    def _build_rfm(self) -> None:
        """
        Compute RFM (Recency, Frequency, Monetary) metrics.

        Definitions:
            Recency   → Days since last purchase
            Frequency → Number of unique transactions
            Monetary  → Total revenue

        Also performs:
            - Quantile-based scoring (R, F, M)
            - Customer segmentation using regex mapping
        """
        self.logger.info("Step: RFM computation")

        # Snapshot date defines "today" for recency calculation
        snapshot_date = self.df["Date"].max() + pd.Timedelta(days=1)

        # Exclude cancelled transactions
        base = self.df[self.df["CancelledInvoice"] == False]

        # --- RFM Core Metrics ---
        recency = (
            base.groupby("CustomerNo")["Date"]
            .agg(lambda x: (snapshot_date - x.max()).days)
            .reset_index(name="Recency")
        )

        frequency = (
            base.groupby("CustomerNo")["TransactionNo"]
            .nunique()
            .reset_index(name="Frequency")
        )

        monetary = (
            base.groupby("CustomerNo")["TotalRevenue"]
            .sum()
            .reset_index(name="Monetary")
        )

        # Merge all RFM components
        rfm = recency.merge(frequency, on="CustomerNo").merge(monetary, on="CustomerNo")

        # Scoring
        # Lower recency = better → reverse scoring
        rfm["R_SCORE"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])

        # Rank frequency to avoid duplicate bin issues
        rfm["F_SCORE"] = pd.qcut(
            rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        )

        rfm["M_SCORE"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

        # --- Segmentation ---
        # Combines R and F scores to assign business-friendly labels
        segs = {
            r"[1-2][1-2]": "Lost",
            r"[1-2][3-4]": "At risk",
            r"[1-2]5": "Can't lose",
            r"3[1-2]": "About to sleep",
            r"33": "Need attention",
            r"41": "Promising",
            r"[3-4][4-5]": "Loyal customer",
            r"51": "New customers",
            r"[4-5][2-3]": "Potential Loyalist",
            r"5[4-5]": "Champion",
        }

        rfm["Segment"] = (
            rfm["R_SCORE"].astype(str) + rfm["F_SCORE"].astype(str)
        ).replace(segs, regex=True)

        # Store result
        self.rfm_table = rfm
        self.metrics["steps_executed"].append("rfm")

    def _build_pareto(self) -> None:
        self.logger.info("Step: Pareto analysis")

        # 1. Aggregate first
        revenue = (
            self.df.groupby("CustomerNo", as_index=False)["TotalRevenue"]
            .sum()
        )

        # 2. Clean data BEFORE sorting
        revenue = revenue[revenue["TotalRevenue"] > 0]

        if revenue.empty:
            self.logger.warning("Pareto skipped: no valid revenue data")
            self.pareto = revenue
            return

        # 3. Sort correctly
        revenue = revenue.sort_values("TotalRevenue", ascending=False).reset_index(drop=True)

        # 4. Cumulative revenue
        revenue["CumRevenue"] = revenue["TotalRevenue"].cumsum()

        total = revenue["TotalRevenue"].sum()
        if total == 0:
            self.logger.warning("Pareto skipped: total revenue is zero")
            self.pareto = revenue
            return

        # 5. Convert to PERCENT SCALE (0–100)
        revenue["CumRevenuePct"] = (revenue["CumRevenue"] / total) * 100

        # 6. Customer cumulative %
        revenue["CumCustomerPct"] = ((revenue.index + 1) / len(revenue)) * 100

        self.pareto = revenue
        self.metrics["steps_executed"].append("pareto")
    def _build_clv(self) -> None:
        """
        Compute Customer Lifetime Value (CLV).

        Formula:
            CLV = AOV * Frequency * Lifespan

        Where:
            AOV      → Average Order Value
            Frequency → Number of transactions
            Lifespan → Active duration in months
        """
        self.logger.info("Step: CLV computation")

        base = self.df[self.df["CancelledInvoice"] == False]

        # Total revenue per customer
        revenue = base.groupby("CustomerNo")["TotalRevenue"].sum()

        # Transaction count
        freq = base.groupby("CustomerNo")["TransactionNo"].nunique()

        # Average order value
        aov = (revenue / freq).reset_index(name="AOV")

        # Customer lifespan (in months)
        lifespan = (
            (
                self.df.groupby("CustomerNo")["Date"].max()
                - self.df.groupby("CustomerNo")["Date"].min()
            ).dt.days.add(1)
            / 30
        ).reset_index(name="Lifespan")

        freq = freq.reset_index(name="Frequency")

        # Merge all components
        clv = aov.merge(freq, on="CustomerNo").merge(lifespan, on="CustomerNo")

        # Final CLV calculation
        clv["CLV"] = clv["AOV"] * clv["Frequency"] * clv["Lifespan"]

        self.clv_table = clv.sort_values(by="CLV", ascending=False)
        self.metrics["steps_executed"].append("clv")

    def _build_cohort(self) -> None:
        """
        Build cohort retention matrix.

        Structure:
            Rows    → CohortMonth (first purchase month)
            Columns → CohortIndex (months since first purchase)
            Values  → Retention rate

        Purpose:
            Analyze customer retention trends over time.
        """
        self.logger.info("Step: Cohort analysis")

        cohort_table = (
            self.df.groupby(["CohortMonth", "CohortIndex"])["CustomerNo"]
            .nunique()
            .unstack(1)
        )

        # Cohort size = number of customers in first period
        cohort_sizes = cohort_table.iloc[:, 0]

        # Normalize to get retention rates
        retention = cohort_table.divide(cohort_sizes, axis=0)

        self.cohort = retention
        self.metrics["steps_executed"].append("cohort")

    # Pipeline Execution


    def run_pipeline(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the full CustomerBehavior pipeline.

        Workflow:
            1. Validate dataset schema
            2. Execute all pipeline steps in defined order

        Returns:
            Dict[str, pd.DataFrame]:
                Dictionary containing:
                    - rfm   → RFM segmentation table
                    - pareto → Pareto distribution table
                    - clv   → Customer lifetime value table
                    - cohort → Cohort retention matrix
        """

        self.logger.info("Starting CustomerBehavior pipeline")

        # Step 1: Schema validation (reusable)
        self._validate_schema()

        # Step 2: Execute pipeline steps sequentially
        for step in self.pipeline_steps:
            step()

        self.logger.info("CustomerBehavior pipeline completed")

        return {
            "rfm": self.rfm_table,
            "pareto": self.pareto,
            "clv": self.clv_table,
            "cohort": self.cohort,
            'tracking_metrics': self.metrics
        }  # type: ignore
