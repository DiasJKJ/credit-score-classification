import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class ClipOutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile, upper_quantile, multiply_by=1.5, replace_with_median=False):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.multiply_by = multiply_by
        self.replace_with_median = replace_with_median
        self.lower_limit = 0
        self.upper_limit = 0

    def fit(self, X, y=None):
        q1, q3 = np.quantile(X, [self.lower_quantile, self.upper_quantile])
        iqr = q3 - q1
        self.lower_limit = q1 - (self.multiply_by * iqr)
        self.upper_limit = q3 + (self.multiply_by * iqr)
        return self

    def transform(self, X):
        if self.replace_with_median:
            return np.where(((X >= self.lower_limit) & (X <= self.upper_limit)), X, np.median(X))
        else:
            return np.clip(X, self.lower_limit, self.upper_limit)
        
import __main__
__main__.ClipOutliersTransformer = ClipOutliersTransformer

app = FastAPI()

outlier_remover = joblib.load("models/OutlierRemover.pkl")
transformer = joblib.load("models/ColumnsTransformers.pkl")

model = CatBoostClassifier()
model.load_model("models/credit_score_prediction_model-cb-v1")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    df_cleaned = df.copy()
    
    df_cleaned[numeric_cols] = outlier_remover.transform(df[numeric_cols])
    
    features_transformed = transformer.transform(df_cleaned)
    
    prediction = model.predict(features_transformed)
    
    if isinstance(prediction, (list, np.ndarray)):
        res = prediction[0]
        if isinstance(res, (list, np.ndarray)):
            res = res[0]
    else:
        res = prediction

    probability = model.predict_proba(features_transformed).tolist()[0]
    
    return {
        "credit_score_class": str(res),
        "probabilities": probability
    }