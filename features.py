import pandas as pd
import logging
import datetime as dt
class FeatureEngineering:

    REQUIRED_COLUMNS = ['TransactionNo','Date','Price','Quantity','CustomerNo']

    def __init__(self, dataframe):
        self.df = dataframe.copy()

        self.quality_metrics = []
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

                # Initialize data quality metrics
        self.feature_metrics = {
            'initial_row_count': len(self.df),
            'rows_affected': {},
            'null_rates_before': {},
            'null_rates_after': {}
        }

    
    def validate_schema(self):
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]

        if missing_columns:
            self.logger.error(f'Missing Required columns: {missing_columns}')
            raise ValueError(f'Required columns missing')
        
        self.logger.info('Schema Validation passed successfully')

    def capture_null_rates(self, stage):
        null_rates = (self.df.isna().mean().round(4).to_dict())

        if stage == 'before':
            self.feature_metrics['null_rates_before'] = null_rates
        else:
            self.feature_metrics['null_rates_after'] = null_rates

    def engineer_features(self):

        self.validate_schema()

        self.capture_null_rates(stage='before')

        # Add total revenue column
        self.df['TotalRevenue'] = self.df['Quantity'] * self.df['Price']

        # Add cancellation flag

        self.df['CancelledInvoice'] = self.df['TransactionNo'].str.startswith('C')

        cancelled_invoices = self.df[self.df['CancelledInvoice']==True]

        number_of_called_invoices = len(cancelled_invoices)

        self.logger.info(f'Total number of cancelled invoices = {number_of_called_invoices}')

        # Add time based features

        self.df['YearMonth'] = self.df['Date'].dt.to_period('M')

        self.df['CohortMonth'] = self.df.groupby('CustomerNo', observed=False)['YearMonth'].transform('min')

        invoice_year = self.df['YearMonth'].dt.year
        invoice_month = self.df['YearMonth'].dt.month
        cohort_year = self.df['CohortMonth'].dt.year
        cohort_month = self.df['CohortMonth'].dt.month

        years_diff = invoice_year - cohort_year
        months_diff = invoice_month - cohort_month

        self.df['CohortIndex'] = ((years_diff * 12) + months_diff+ 1)

        return self.df
