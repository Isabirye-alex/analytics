import pandas as pd
import os
from sqlalchemy import create_engine
from reusables.reusable_functions import ReusableFunctions
from typing import Dict, Any


class DataIngestor:
    """
    Handles data ingestion from CSV files and persistence to PostgreSQL.

    Responsibilities:
        - Load raw CSV data into a DataFrame
        - Persist processed DataFrames to PostgreSQL
        - Track ingestion metrics
    """

    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "rows_ingested": None,
            "source_path": None,
        }
        self.logger = ReusableFunctions.setup_logger(self.__class__.__name__)

    def fetch_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Raw loaded data.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        df = pd.read_csv(file_path)

        self.metrics["rows_ingested"] = len(df)
        self.metrics["source_path"] = file_path

        self.logger.info(f"Ingested {len(df)} rows from {file_path}")

        return df

    def save_to_postgres(
        self,
        df: pd.DataFrame,
        table_name: str,
        connection_string: str,
    ) -> None:
        """
        Persist a DataFrame to a PostgreSQL table.

        Converts Period and category columns to string before writing
        since PostgreSQL does not support these dtypes natively.

        Args:
            df (pd.DataFrame): Data to persist.
            table_name (str): Target table name.
            connection_string (str): SQLAlchemy connection string.

        Raises:
            RuntimeError: If the database write fails.
        """

        engine = create_engine(connection_string)

        try:
            df = df.copy()

            # PostgreSQL does not support Period dtype
            for col in df.columns:
                if pd.api.types.is_period_dtype(df[col]):  # type: ignore
                    df[col] = df[col].astype(str)

            # PostgreSQL does not support category dtype
            for col in df.select_dtypes(include=["category"]).columns:
                df[col] = df[col].astype(str)

            df.to_sql(
                table_name,
                engine,
                if_exists="replace",
                index=False,
            )

            self.logger.info(f"Saved {len(df)} rows to table '{table_name}'")

        except Exception as e:
            self.logger.error(f"Failed to save '{table_name}' to PostgreSQL: {e}")
            raise RuntimeError(f"Database write failed for table '{table_name}'") from e
