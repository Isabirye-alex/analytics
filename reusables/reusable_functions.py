import logging
import pandas as pd
from typing import Dict


class ReusableFunctions:
    """
    Utility class for reusable data processing functions.
    All methods are stateless and reusable across projects.
    """

    @staticmethod
    def setup_logger(name: str) -> logging.Logger:
        """
        Create and configure a logger.

        Args:
            name (str): Logger name.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(name)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    @staticmethod
    def validate_schema(
        df: pd.DataFrame, required_columns: list, logger: logging.Logger
    ) -> None:
        """
        Validate required columns exist in dataframe.

        Args:
            df (pd.DataFrame): Dataset to validate
            required_columns (list): Required column names
            logger (logging.Logger): Logger instance

        Raises:
            ValueError: If columns are missing
        """
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Schema validation passed")

    @staticmethod
    def capture_null_rates(
        df: pd.DataFrame, quality_metrics: Dict, stage: str, logger: logging.Logger
    ) -> None:
        """
        Capture null rates for dataframe.

        Args:
            df (pd.DataFrame): Dataset
            quality_metrics (Dict): Metrics dictionary to update
            stage (str): 'before' or 'after'
            logger (logging.Logger): Logger instance
        """
        null_rates = df.isna().mean().round(4).to_dict()

        if stage == "before":
            quality_metrics["null_rates_before"] = null_rates
        elif stage == "after":
            quality_metrics["null_rates_after"] = null_rates
        else:
            raise ValueError("Stage must be 'before' or 'after'")

        logger.info(f"Captured null rates ({stage})")
