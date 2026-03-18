import pandas as pd
import logging


class CustomerBehavior:

    REQUIRED_COLUMNS = [
        "TotalRevenue",
        "CohortIndex",
        "CohortMonth",
        "CustomerNo",
    ]

    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.rfm_table = None
        self.clv_table = None
        self.pareto = None
        self.cohort = None
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(ascitime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.quality_metrics = []

    def validate_schema(self):
        missing_columns = [
            col for col in self.REQUIRED_COLUMNS if col not in self.df.columns
        ]

        if missing_columns:
            self.logger.error(f"Missing required columns : {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.logger.info(f"Schema validation passed successfully")

    def build_rfm(self):
        snapshot_date = self.df["Date"].max() + pd.Timedelta(days=1)

        recency = (
            self.df[self.df["CancelledInvoice"] == False]
            .groupby("CustomerNo", observed=False)["Date"]
            .agg(lambda x: (snapshot_date - x.max()).days)
            .reset_index()
            .rename(columns={"Date": "Recency"})
        )

        frequency = (
            (self.df[self.df["CancelledInvoice"] == False])
            .groupby("CustomerNo", observed=False)["TransactionNo"]
            .nunique()
            .reset_index()
            .rename(columns={"TransactionNo": "Frequency"})
        )
        monetary = (
            (self.df[self.df["CancelledInvoice"] == False])
            .groupby("CustomerNo", observed=False)["TotalRevenue"]
            .sum()
            .reset_index()
            .rename(columns={"TotalRevenue": "Monetary"})
        )
        rfm_table = recency.merge(frequency, on="CustomerNo")
        rfm_table = rfm_table.merge(monetary, on="CustomerNo")
        rfm_table["R_SCORE"] = pd.qcut(rfm_table["Recency"], 5, labels=[5, 4, 3, 2, 1])
        rfm_table["F_SCORE"] = pd.qcut(
            rfm_table["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        )
        rfm_table["M_SCORE"] = pd.qcut(rfm_table["Monetary"], 5, labels=[1, 2, 3, 4, 5])

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
        rfm_table["Segment"] = (
            rfm_table["R_SCORE"].astype(str) + rfm_table["F_SCORE"].astype(str)
        ).replace(segs, regex=True)

        self.rfm_table = rfm_table

        return self.rfm_table

    def build_pareto(self):
        customer_revenue = (
            self.df.groupby("CustomerNo", observed=False)["TotalRevenue"]
            .sum()
            .reset_index()
            .sort_values(ascending=False, by="TotalRevenue")
        )
        customer_revenue = customer_revenue[customer_revenue["TotalRevenue"] > 0]
        customer_revenue["CumRevenue"] = customer_revenue["TotalRevenue"].cumsum()
        customer_revenue["CumRevenuePct"] = (
            customer_revenue["CumRevenue"] / customer_revenue["TotalRevenue"].sum()
        )
        customer_revenue = customer_revenue.reset_index(drop=True)
        customer_revenue["CumCustomerPct"] = (customer_revenue.index + 1) / len(
            customer_revenue
        )
        self.pareto = customer_revenue
        return self.pareto

    def build_clv(self):
        total_customer_revenue = (
            self.df[self.df["CancelledInvoice"] == False]
            .groupby("CustomerNo")["TotalRevenue"]
            .sum()
        )
        total_customer_transactions = (
            self.df[self.df["CancelledInvoice"] == False]
            .groupby("CustomerNo")["TransactionNo"]
            .nunique()
        )
        AOV = total_customer_revenue / total_customer_transactions
        AOV = AOV.reset_index()
        AOV.columns = ["Customer", "AOV"]

        lifespan = (
            (
                self.df.groupby("CustomerNo")["Date"].max()
                - self.df.groupby("CustomerNo")["Date"].min()
            ).dt.days.add(1)
        ) / 30
        lifespan = lifespan.reset_index()
        lifespan.columns = ["Customer", "Lifespan"]

        frequency = (
            self.df[self.df["CancelledInvoice"] == False]
            .groupby("CustomerNo", observed=False)["TransactionNo"]
            .nunique()
        )
        frequency = frequency.reset_index()
        frequency.columns = ["Customer", "Frequency"]

        clv_table = pd.merge(AOV, frequency, on="Customer")
        clv_table = pd.merge(clv_table, lifespan, on="Customer")
        clv_table = clv_table.sort_values(by="AOV", ascending=False)
        clv_table["CLV"] = (
            clv_table["AOV"] * clv_table["Frequency"] * clv_table["Lifespan"]
        )
        self.clv_table = clv_table

        return self.clv_table

    def build_cohort(self):
        cohort_table = (
            self.df.groupby(["CohortMonth", "CohortIndex"])["CustomerNo"]
            .nunique()
            .unstack(1)
            .rename(columns={"CustomerNo": "Customers"})
        )

        cohort_sizes = cohort_table.iloc[:, 0]
        retention_table = cohort_table.divide(cohort_sizes, axis=0)
        self.cohort = retention_table
        return self.cohort
