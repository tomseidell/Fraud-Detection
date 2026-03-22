import pandas as pd
from typing import Self
from pandas import DataFrame
from pathlib import Path


class Loader:
    def __init__(self):
        root = Path(__file__).parent.parent.parent 
        self.data_train_id_path = root / "data" / "raw" / "train_identity.csv"
        self.data_train_trans_path = root / "data" / "raw" / "train_transaction.csv"
        self.data_test_id_path = root / "data" / "raw" / "test_identity.csv"
        self.data_test_trans_path = root / "data" / "raw" / "test_transaction.csv"

    def _load_csv_data(self):
        '''
        This function loads both our dataset files for train and test and saves them as class variables
        '''
        self.train_id_data = pd.read_csv(self.data_train_id_path)
        self.train_trans_data = pd.read_csv(self.data_train_trans_path)
        self.test_id_data = pd.read_csv(self.data_test_id_path)
        self.test_trans_data = pd.read_csv(self.data_test_trans_path)

    def _merge_train_data(self) -> DataFrame:
        '''
        This method merges both train datasets together
        '''

        df = pd.merge(self.train_trans_data, self.train_id_data, how="left", on="TransactionID") # left (trans) = dominant df, transaction id = identifier
        return df

    def _merge_test_data(self):
        '''
        This method merges both test datasets together
        '''
        df = pd.merge(self.test_trans_data, self.test_id_data, how="left", on="TransactionID")
        df.columns = df.columns.str.replace('-', '_') # column names differ from train columns. Make them have equal names
        return df

    
    def load(self, include_test=False)->Self:
        self._load_csv_data()
        self.df = self._merge_train_data()
        if include_test:
            self.test_df = self._merge_test_data()

        return self
