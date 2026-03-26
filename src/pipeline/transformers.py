import pandas as pd 
import numpy as np 
from pandas import DataFrame
from typing import Self
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    This class drops given columns from our dataset.
    This removes noise and helps the model to improve.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: DataFrame, y=None) -> Self:
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        return X.drop(columns=self.columns, errors="ignore")


class EmailTransformer(BaseEstimator, TransformerMixin):
    """
    This class encodes the "R_emaildomain" column to numbers 1-6.
    It extracts the 5 most used emails from the train sets and assigns 
    a number (1-5) to each. Each other email will be represented by the number 6
    """
    
    def __init__(self, number_of_mail_provider: int):
        self.number_of_mail_provider = number_of_mail_provider
        self.email_cols = ["R_emaildomain", "P_emaildomain"]

    def fit(self, X, y=None):
        self.email_mappers_ = {}
        for col in self.email_cols:
            if col not in X.columns:
                continue
            top_n = X[col].value_counts().head(self.number_of_mail_provider)
            # create nested object containing all n email domains for column 
            self.email_mappers_[col] = {domain: i+1 for i, domain in enumerate(top_n.index)}
        return self

    def transform(self, X):
        X = X.copy()
        for col, mapper in self.email_mappers_.items():
            X[col] = X[col].map(mapper).fillna(len(mapper) + 1).astype(int)
        return X
    

class OHETransformer(BaseEstimator, TransformerMixin):
    """
    This class receives an array of columns which should be one hot encoded.
    The columns are being saved in the fit step and afterwards used in tranform.
    If a new value is being seen in transform, that does not belong to any of the columns
    saved in the class, all created columns will have the value 0
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X:DataFrame, y=None)-> Self:
        self.ohe_categories_ = {
            col: X[col].dropna().unique()
            for col in self.columns
        }
        return self
    
    def transform(self, X:DataFrame) -> DataFrame:
        X = X.copy()
        for col, categories in self.ohe_categories_.items():
            X[col] = pd.Categorical(X[col], categories=categories) 
        X = pd.get_dummies(X, columns=list(self.ohe_categories_.keys()))
        bool_cols = X.select_dtypes(include="bool").columns
        X[bool_cols] = X[bool_cols].astype(int)  
        return X


class TimeTransformer(BaseEstimator, TransformerMixin):
    """
    This class transforms multiple time related columns with a static formula.
    """

    def fit(self, X: DataFrame, y=None) -> Self:
        return self 

    def transform(self, X:DataFrame) -> DataFrame:
        X = X.copy()
        X["DT_day"] = X["TransactionDT"] // 86400
        X["DT_hour"] = (X["TransactionDT"] // 3600) % 24
        X["DT_weekday"] = X["DT_day"] % 7

        X["DT_hour_sin"] = np.sin(2 * np.pi * X["DT_hour"] / 24)
        X["DT_hour_cos"] = np.cos(2 * np.pi * X["DT_hour"] / 24)

        X["DT_weekday_sin"] = np.sin(2 * np.pi * X["DT_weekday"] / 7)
        X["DT_weekday_cos"] = np.cos(2 * np.pi * X["DT_weekday"] / 7)


        X = X.drop(["DT_day", "TransactionDT", "DT_weekday", "DT_hour"], axis=1)
        return X


class OrdinalTransformer(BaseEstimator, TransformerMixin):
    """
    Ordinal encodes all remaining object columns.
    NaN values are kept as NaN so XGBoost can handle them natively.
    Unknown categories (seen in test but not in train) are encoded as -1.
    """

    def fit(self, X: DataFrame, y=None) -> Self:
        self.obj_cols_ = X.select_dtypes(include="object").columns.tolist()
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=np.nan
        )
        self.encoder_.fit(X[self.obj_cols_])
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        X[self.obj_cols_] = self.encoder_.transform(X[self.obj_cols_])
        return X
    

class FrequencyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=list[str]):
        self.columns = columns
        self.freq_map = {}

    def fit(self, X: DataFrame, y=None) -> Self:
        for col in self.columns:
            self.freq_map[col] = X[col].value_counts(normalize=True).to_dict()
        return self
    
    def transform(self, X:DataFrame) ->DataFrame:
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.freq_map[col])
        return X


class UidTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: DataFrame, y=None) -> Self:
        return self  

    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        
        X["uid"] = X[self.columns[0]].astype(str)
        
        for col in self.columns[1:]:
            X["uid"] += "_" + X[col].fillna("unknown").astype(str)
        
        return X