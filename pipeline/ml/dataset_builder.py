import pandas as pd
import datetime as dt

class DatasetBuilder:
    """
    Combines RFM and CLV tables into a single customer-level dataset
    for machine learning.

    Responsibilities:
        - Merge RFM and CLV features
        - Ensure correct join keys
        - Prepare dataset for downstream ML pipeline
    """

    def __init__(self, rfm: pd.DataFrame, clv: pd.DataFrame, dataframe):
        """
        Initialize DatasetBuilder.

        Args:
            rfm (pd.DataFrame): RFM feature table
            clv (pd.DataFrame): CLV feature table
        """

        self.rfm = rfm.copy()
        self.clv = clv.copy()
        self.df = dataframe.copy()
        self.dataset: pd.DataFrame | None = None

    def build(self) -> pd.DataFrame:
        """
        Merge RFM and CLV datasets.

        Returns:
            pd.DataFrame: Combined dataset
        """

        #  Validate required key
        if "CustomerNo" not in self.rfm.columns:
            raise ValueError("CustomerNo missing in RFM table")

        if "CustomerNo" not in self.clv.columns:
            raise ValueError("CustomerNo missing in CLV table")
        self.clv = self.clv.drop(columns=['Frequency'])
        #  Merge datasets
        dataset = pd.merge(
            self.rfm,
            self.clv,
            on="CustomerNo",
            how="inner",  # ensures only matching customers
        )

        # Revenue in last 90 days vs previous 90 days
        snapshot = self.df["Date"].max()

        recent = self.df[self.df["Date"] >= snapshot - pd.Timedelta(days=90)]
        prior = self.df[(self.df["Date"] >= snapshot - pd.Timedelta(days=180)) & 
                (self.df["Date"] < snapshot - pd.Timedelta(days=90))]

        recent_rev = recent.groupby("CustomerNo")["TotalRevenue"].sum()
        prior_rev = prior.groupby("CustomerNo")["TotalRevenue"].sum()

        trend = (recent_rev - prior_rev).reset_index()
        trend.columns = ["CustomerNo", "RevenueTrend"]

        # Negative = declining spend = churn signal

        MINIMUM_DAYS = 60

        purchase_date = self.df[self.df['CancelledInvoice']==False].sort_values('Date').groupby('CustomerNo')['Date']
        avg_gap = purchase_date.apply(lambda x: x.diff().dt.days.mean()).reset_index()
        avg_gap.columns = ['CustomerNo', 'AvgGapDays']

        dataset = pd.merge(dataset, avg_gap, on='CustomerNo', how='inner')
        dataset = dataset.merge(trend, on='CustomerNo', how='left')

        dataset['ChurnThreshold'] = (dataset['AvgGapDays']*2).clip(lower=MINIMUM_DAYS)

        dataset['Churn'] = (dataset['Recency'] > (dataset['ChurnThreshold'])).astype('Int64')
        dataset = dataset.drop(columns={'AOV','Monetary'})
        # print(
        #     dataset.groupby("Churn")[
        #         ["Frequency", "Lifespan", "CLV", "AvgGapDays"]
        #     ].mean()
        # )
        # print('Model Testing....................................................')
        # print(
        #     dataset.groupby("Churn")[
        #         ["Frequency", "Lifespan", "CLV", "RevenueTrend"]
        #     ].mean()
        # )

        # These are my most valuable at-risk customers
        at_risk = dataset[
            (dataset["CLV"] > dataset["CLV"].quantile(0.75)) &
            (dataset["RevenueTrend"] < 0) &
            (dataset["Churn"] == 1)
        ][["CustomerNo", "CLV", "RevenueTrend", "Frequency"]].sort_values(
            "CLV", ascending=False
        )

        # print(at_risk.head(10))
        #  Store result
        self.dataset = dataset

        return dataset
