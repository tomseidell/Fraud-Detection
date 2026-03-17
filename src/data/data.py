import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from pathlib import Path

class Data:
    def __init__(self):
        root = Path(__file__).parent.parent.parent 
        self.data_train_id_path = root / "data" / "raw" / "train_identity.csv"
        self.data_train_trans_path = root / "data" / "raw" / "train_transaction.csv"
        self.data_test_id_path = root / "data" / "raw" / "test_identity.csv"
        self.data_test_trans_path = root / "data" / "raw" / "test_transaction.csv"
        self.label_encoders = {} # memory for label encoding 

    def load_csv_data(self):
        '''
        This function loads both our dataset files and saves them as class variables
        '''
        self.train_id_data = pd.read_csv(self.data_train_id_path)
        self.train_trans_data = pd.read_csv(self.data_train_trans_path)
        self.test_id_data = pd.read_csv(self.data_test_id_path)
        self.test_trans_data = pd.read_csv(self.data_test_trans_path)


    def merge_csv_data(self) -> DataFrame:
        '''
        This merges both datasets loaded above
        '''

        df = pd.merge(self.train_trans_data, self.train_id_data, how="left", on="TransactionID") # left (trans) = dominant df, transaction id = identifier
        self.df = df
        return df

    def merge_csv_test_data(self):
        df = pd.merge(self.test_trans_data, self.test_id_data, how="left", on="TransactionID")
        df.columns = df.columns.str.replace('-', '_')
        self.df = df

    def perform_feature_engineering(self):

        def group_emails():
            '''
            This helper method transforms the "R_emaildomain" column in our dataset.
            I assigns numeric values (1-5) to the top 5 email provider.
            As this column does have many nullvalues (80%), we assign another numeric value (6) to all null and uncommon (not top 5) domains
            '''

            top5_email_categories = self.df["R_emaildomain"].value_counts().head(5)

            email_mapper = {}
            for i, domain in enumerate(top5_email_categories.index.tolist()):
                email_mapper[domain] = i+1

            self.df["R_emaildomain"] = self.df["R_emaildomain"].map(email_mapper).fillna(6).astype(int)

        def one_hot_encoding_helper(column:str, prefix:str):
            '''
            This helper method performs one hot encoding to a given column and adds a prefix to all generated columns
            '''
            self.df = pd.get_dummies(data = self.df, columns=[column], prefix=prefix)

        def time_column_transformer():
            '''
            This helper method transforms our "TransactionDT" column into 2 new, more valuable columns
            - DT_weekday: 0-6 (represents a certain day of a week)
            - DT_hour: 0-23 (represents a certain hour of a day)
            '''
            self.df["DT_day"] = self.df["TransactionDT"] // 86400
            self.df["DT_hour"] = (self.df["TransactionDT"] // 3600) % 24
            self.df["DT_weekday"] = self.df["DT_day"] % 7

            self.df = self.df.drop(["DT_day", "TransactionDT"], axis=1)

        def label_encoding():
            '''
            This helper method performs label encoding on all our string columns.
            We keep the NaN columns and do not map those to a dedicated number. 
            '''
            cat_cols = self.df.select_dtypes(include="str").columns

            for col in cat_cols:
                mask = self.df[col].notna()

                if col not in self.label_encoders:
                    le = LabelEncoder()
                    le.fit(self.df.loc[mask, col])
                    self.label_encoders[col] = le

                le = self.label_encoders[col]
                encoded = pd.Series(index=self.df.index, dtype="float64")
                known = set(le.classes_)
                safe_mask = mask & self.df[col].isin(known)
                encoded[safe_mask] = le.transform(self.df.loc[safe_mask, col])
                encoded[mask & ~self.df[col].isin(known)] = -1
                self.df[col] = encoded

        group_emails()

        one_hot_encoding_helper(column="ProductCD", prefix="ProductCD")
        one_hot_encoding_helper(column="card4", prefix="card4")
        one_hot_encoding_helper(column="card6", prefix="card6")
        one_hot_encoding_helper(column="DeviceType", prefix="DeviceType")

        time_column_transformer()

        label_encoding()
    
    def split_data(self):
        '''
        This method transforms the merged dataset into 3 subsets:
        - train (70%)
        - val (15%)
        - test (15%)
        '''
        y = self.df['isFraud']
        X = self.df.drop(columns=["isFraud"])

        X_train, X_val_test, y_train, y_val_test = train_test_split(X,y, test_size=0.3, random_state=40, stratify=y)
        # stratify = y, make sure equal share of y = 1 for all subsets

        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=40, stratify=y_val_test)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def prepare_data(self, test_data: bool = False):
        self.load_csv_data()

        self.merge_csv_data()
        self.perform_feature_engineering()
        self.split_data()
        self.train_columns = self.X_train.columns.tolist()

        if test_data:
            self.merge_csv_test_data()
            self.perform_feature_engineering()
            return self.df.reindex(columns=self.train_columns, fill_value=0)
        else:
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
