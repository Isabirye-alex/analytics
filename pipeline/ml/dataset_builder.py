import pandas as pd


class DatasetBuilder:
    """
    Combines RFM and CLV tables into a single customer-level dataset
    for machine learning.

    Responsibilities:
        - Merge RFM and CLV features
        - Ensure correct join keys
        - Prepare dataset for downstream ML pipeline
    """

    def __init__(self, rfm: pd.DataFrame, clv: pd.DataFrame):
        """
        Initialize DatasetBuilder.

        Args:
            rfm (pd.DataFrame): RFM feature table
            clv (pd.DataFrame): CLV feature table
        """

        self.rfm = rfm.copy()
        self.clv = clv.copy()
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

        dataset['Churn'] = (dataset['Recency']>90).astype('Int64')
        dataset = dataset.drop(columns={'AOV','Monetary'})

        #  Store result
        self.dataset = dataset

        return dataset
