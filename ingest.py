import pandas as pd
import os
import logging

class DataIngestor:

    def __init__(self, path):
        self.source_path = path
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

    