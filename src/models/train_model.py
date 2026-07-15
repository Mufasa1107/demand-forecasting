from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.features.build_features import build_feature_frame


def train_lightgbm_forecasts(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Train a LightGBM regressor per SKU and return forecast-ready predictions."""
    feature_df = build_feature_frame(df, horizon=horizon)
    feature_df["target"] = feature_df["demand"]

    feature_columns = [
        "day_of_week",
        "month",
        "is_weekend",
        "promo_flag",
        "price",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_7",
    ]

    predictions: List[dict] = []
    for sku in feature_df["sku"].unique():
        sku_df = feature_df[feature_df["sku"] == sku].copy()
        X_train = sku_df[feature_columns]
        y_train = sku_df["target"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        pred = model.predict(X_train)
        sku_df = sku_df.assign(prediction=pred)
        sku_df = sku_df[["date", "region", "store_id", "category", "sku", "prediction", "demand"]]
        predictions.append(sku_df)

    return pd.concat(predictions, ignore_index=True)
