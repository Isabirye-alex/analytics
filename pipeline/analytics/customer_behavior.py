import pandas as pd
import logging


class CustomerBehavior:

    REQUIRED_COLUMNS = [
        'TotalRevenue',
        'CohortIndex',
        'CohortMonth',
        'CustomerNo',
    ]

    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.rfm_table = None
        self.clv_table = None
        self.pareto = None
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO, format='%(ascitime)s - %(name)s - %(levelname)s - %(message)s')
        self.quality_metrics = []

    def validate_schema(self):
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]

        if missing_columns:
            self.logger.error(f'Missing required columns : {missing_columns}')
            raise ValueError(f'Missing required columns: {missing_columns}')
        
        self.logger.info(f'Schema validation passed successfully')
    
    def build_rfm(self):
        snapshot_date = self.df['Date'].max() + pd.Timedelta(days=1)

        recency = (self.df[self.df['CancelledInvoice']==False].groupby('CustomerNo', observed=False)['Date'].agg(
            lambda x: (snapshot_date - x.max()).days
        ).reset_index().rename(columns = {'Date':'Recency'})
        )

        frequency = (self.df[self.df['CancelledInvoice']==False]).groupby('CustomerNo', observed=False)['TransactionNo'].nunique().reset_index().rename(columns={'TransactionNo':'Frequency'})
        monetary = (self.df[self.df['CancelledInvoice']==False]).groupby('CustomerNo', observed=False)['TotalRevenue'].sum().reset_index().rename(columns={'TotalRevenue':'Monetary'})
        rfm_table = recency.merge(frequency, on='CustomerNo')
        rfm_table = rfm_table.merge(monetary, on='CustomerNo')
        self.rfm_table = rfm_table

        return self.rfm_table
