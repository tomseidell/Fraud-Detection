import pandas as pd 
import numpy as np 
from pandas import DataFrame
from typing import Self
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

class DropColumnsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: DataFrame, y=None) -> Self:
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        return X.drop(columns=self.columns, errors="ignore")


class EmailTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X:DataFrame, y=None) -> Self:
        top5_email_categories = X["R_emaildomain"].value_counts().head(5)

        self.email_mapper_ = {} # save domain and domain specific number (1-5)
        for i, domain in enumerate(top5_email_categories.index.tolist()):
            self.email_mapper_[domain] = i+1
        return self
    
    def transform(self, X:DataFrame) -> DataFrame:
        X = X.copy()
        X["R_emaildomain"] = X["R_emaildomain"].map(self.email_mapper_).fillna(6).astype(int)
        return X
    

class OHETransformer(BaseEstimator, TransformerMixin):

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
