import pandas as pd
from typing import Self
from pandas import DataFrame
from pathlib import Path

class Data:
    def __init__(self):
        root = Path(__file__).parent.parent.parent 
        self.data_train_id_path = root / "data" / "raw" / "train_identity.csv"
        self.data_train_trans_path = root / "data" / "raw" / "train_transaction.csv"
        self.data_test_id_path = root / "data" / "raw" / "test_identity.csv"
        self.data_test_trans_path = root / "data" / "raw" / "test_transaction.csv"

    def load_csv_data(self):
        '''
        This function loads both our dataset files for train and test and saves them as class variables
        '''
        self.train_id_data = pd.read_csv(self.data_train_id_path)
        self.train_trans_data = pd.read_csv(self.data_train_trans_path)
        self.test_id_data = pd.read_csv(self.data_test_id_path)
        self.test_trans_data = pd.read_csv(self.data_test_trans_path)

    def merge_csv_data(self) -> DataFrame:
        '''
        This method merges both train datasets together
        '''

        df = pd.merge(self.train_trans_data, self.train_id_data, how="left", on="TransactionID") # left (trans) = dominant df, transaction id = identifier
        self.df = df
        return df

    def merge_csv_test_data(self):
        '''
        This method merges both test datasets together
        '''
        df = pd.merge(self.test_trans_data, self.test_id_data, how="left", on="TransactionID")
        df.columns = df.columns.str.replace('-', '_') # column names differ from train columns. Make them have equal names
        self.test_df = df

    def fit(self, X_train:DataFrame):
        '''
        This helper method transforms the "R_emaildomain" column in our dataset.
        I assigns numeric values (1-5) to the top 5 email provider.
        As this column does have many nullvalues (80%), we assign another numeric value (6) to all null and uncommon (not top 5) domains
        '''

        # dynamically assign top 5 categories to given dataframe
        top5_email_categories = X_train["R_emaildomain"].value_counts().head(5)


        self.email_mapper = {} # save domain and domain specific number (1-5)
        for i, domain in enumerate(top5_email_categories.index.tolist()):
            self.email_mapper[domain] = i+1

        self.ohe_categories = {
            "ProductCD": X_train["ProductCD"].dropna().unique(),
            "card4":     X_train["card4"].dropna().unique(),
            "card6":     X_train["card6"].dropna().unique(),
            "DeviceType": X_train["DeviceType"].dropna().unique()
        }

    def transform(self, df:DataFrame) -> DataFrame:

        def time_column_transformer(df:DataFrame):
            '''
            This helper method transforms our "TransactionDT" column into 2 new, more valuable columns
            - DT_weekday: 0-6 (represents a certain day of a week)
            - DT_hour: 0-23 (represents a certain hour of a day)
            '''
            df["DT_day"] = df["TransactionDT"] // 86400
            df["DT_hour"] = (df["TransactionDT"] // 3600) % 24
            df["DT_weekday"] = df["DT_day"] % 7

            df = df.drop(["DT_day", "TransactionDT"], axis=1)
            return df
        
        df = time_column_transformer(df=df)

        df["R_emaildomain"] = df["R_emaildomain"].map(self.email_mapper).fillna(6).astype(int)

        df = df.drop(columns=["TransactionID"])

        for col, categories in self.ohe_categories.items():
            df[col] = pd.Categorical(df[col], categories=categories) 
        
        df = pd.get_dummies(df, columns=list(self.ohe_categories.keys()))

        return df
  
    def split_data(self):
        y = self.df['isFraud']
        X = self.df.drop(columns=["isFraud"])

        split_idx = int(len(X) * 0.7)

        self.X_train = X.iloc[:split_idx]
        self.y_train = y.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_test = y.iloc[split_idx:]

    def prepare_data(self, test_data: bool = False):
        """
        performs feature loading, merging and transforming for both train and test data
        """
        self.load_csv_data()

        self.merge_csv_data()
        self.split_data()
        self.fit(self.X_train)
        self.X_train = self.transform(df=self.X_train)
        self.train_columns = self.X_train.columns.tolist()
        self.X_test = self.transform(df=self.X_test)
        self.X_test = self.X_test.reindex(columns=self.train_columns, fill_value=0)

        if test_data: 
            self.merge_csv_test_data()
            X_test = self.transform(self.test_df)
            return X_test.reindex(columns=self.train_columns, fill_value=0) # reorder column sequence, to match train data
        else:
            return self.X_train, self.X_test, self.y_train, self.y_test


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
