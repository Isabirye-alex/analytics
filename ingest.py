import pandas as pd
import os
import logging
from sqlalchemy import create_engine


class DataIngestor:

    def __init__(self, path):
        self.source_path = path
        # self.engine = create_engine(connection_string)
        self.metrics = []
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def fetch_data_from_csv(self):
        if not os.path.exists(self.source_path):

            raise FileNotFoundError(f"File Not found at {self.source_path}")
        return pd.read_csv(self.source_path)

    # def save_to_postgress(self, df, tablename):
    #     """
    #     Stores a specific dataframe to PostgreSQL.
    #     Converts unsupported types (Period, category) to string automatically.
    #     """
    #     try:
    #         # Convert Period columns to string
    #         for col in df.columns:
    #             if pd.api.types.is_period_dtype(df[col]): # type: ignore
    #                 df[col] = df[col].astype(str)

    #         # Convert category columns to string
    #         cat_cols = df.select_dtypes(include=['category']).columns
    #         for col in cat_cols:
    #             df[col] = df[col].astype(str)

    #         # Save to PostgreSQL
    #         df.to_sql(
    #             tablename,
    #             self.engine,
    #             if_exists='replace',
    #             index=False
    #         )
    #         print('Successfully saved to db')
    #     except Exception as e:
    #         print(f'Error saving to db: {e}')
    #         import traceback
    #         traceback.print_exc()
