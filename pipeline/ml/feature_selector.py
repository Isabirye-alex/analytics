import pandas as pd
from typing import List


class FeatureSelector:
    """
    Prepares a clean feature matrix for ML while avoiding leakage.
    """

    # Columns that should NEVER go into the model
    DROP_COLUMNS: List[str] = [
        "CustomerNo",
        "TransactionNo",
        "ProductNo",
        "ProductName",
        "Country",
        "Date",
        "YearMonth",
        "CohortMonth",
        "Segment",
    ]

    # Columns that could leak future info
    LEAKAGE_COLUMNS: List[str] = ["Recency", "Lifespan", "CLV", 'AvgGapDays', 'ChurnThreshold']
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        

    def select(self) -> pd.DataFrame:
        """
        Returns clean numeric features without leakage or IDs.
        """
        df = self.df.copy()

        # df = df.drop(columns=[self.label], errors="ignore")

        #  Remove identifiers
        df = df.drop(columns=self.DROP_COLUMNS, errors="ignore")

        #  Remove leakage columns
        df = df.drop(columns=self.LEAKAGE_COLUMNS, errors="ignore")

        #  Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        #  Keep only numeric features
        df = df.select_dtypes(include=["int64", "float64"])

        #  Handle missing values
        df = df.fillna(0)

        return df
