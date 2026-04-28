import pandas as pd 
import numpy as np 
from pandas import DataFrame
from typing import Self
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA

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
    Derives time features from the raw TransactionDT (seconds offset) column.
    Hour and weekday are encoded as sin/cos pairs to keep cyclic nature,
    then the original column is dropped.
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
    """
    This class performs frequency encoding to columns with high cardinality.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.freq_map = {}

    def fit(self, X: DataFrame, y=None) -> Self:
        for col in self.columns:
            if col not in X.columns:  # ← Fix
                continue
            self.freq_map[col] = X[col].value_counts(normalize=True).to_dict()
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        for col in self.freq_map:   
            if col not in X.columns:
                continue
            X[col] = X[col].map(self.freq_map[col])
        return X


class UidTransformer(BaseEstimator, TransformerMixin):
    """
    This class creates a single uid, which should approximate an individual user identifier.
    The uid is been dynamically created based on the columns which are being passed.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: DataFrame, y=None) -> Self:
        X = X.copy()
        uid = X[self.columns[0]].astype(str)
        for col in self.columns[1:]:
            uid += "_" + X[col].fillna("unknown").astype(str)
        self.uid_freq_ = uid.value_counts(normalize=True).to_dict()
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        uid = X[self.columns[0]].astype(str)
        for col in self.columns[1:]:
            uid += "_" + X[col].fillna("unknown").astype(str)
        X["uid"] = uid.map(self.uid_freq_).fillna(0).astype(float)
        return X

    
class AmountTransformer(BaseEstimator, TransformerMixin):
    """
    This class splits the TransactionAmt column into 4 new columns
    """

    def fit(self, X: DataFrame, y=None) -> Self:
        return self 
    
    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()

        X["Trans_amt_dollar"] = X["TransactionAmt"].round(2)
        X["Trans_amt_cents"]  = (X["TransactionAmt"] % 1 * 100).round(0).astype(int)
        X["Trans_amt_log"]    = np.log1p(X["TransactionAmt"])
        X["Trans_amt_isround"] = (X["TransactionAmt"] % 1 == 0).astype(int)

        return X


class PCATransformer(BaseEstimator, TransformerMixin):
    """
    This class performs PCA on the V columns (300+) of our dataset.
    The amount of components in the PCA are being dynamically defined as an input.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X: DataFrame, y=None) -> Self:
        self.v_cols = [col for col in X.columns if col.startswith("V")]
        self.pca = PCA(n_components=self.n_components)  
        self.pca.fit(X[self.v_cols].fillna(0))
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        pca_result = self.pca.transform(X[self.v_cols].fillna(0))
        for i in range(self.n_components):
            X[f"V_pca_{i+1}"] = pca_result[:, i]
        X = X.drop(columns=self.v_cols)
        return X