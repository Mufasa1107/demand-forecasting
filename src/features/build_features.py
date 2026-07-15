from __future__ import annotations

import pandas as pd


def build_feature_frame(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Create lagged and calendar features for forecasting models."""
    feature_df = df.sort_values(["sku", "date"]).copy()
    feature_df["day_of_week"] = feature_df["date"].dt.dayofweek
    feature_df["month"] = feature_df["date"].dt.month
    feature_df["is_weekend"] = feature_df["day_of_week"].isin([5, 6]).astype(int)

    feature_df["lag_7"] = feature_df.groupby("sku")["demand"].shift(7)
    feature_df["rolling_mean_7"] = (
        feature_df.groupby("sku")["demand"].transform(lambda s: s.shift().rolling(7, min_periods=7).mean())
    )
    feature_df["rolling_std_7"] = (
        feature_df.groupby("sku")["demand"].transform(lambda s: s.shift().rolling(7, min_periods=7).std())
    )

    feature_df = feature_df.sort_values(["sku", "date"]).reset_index(drop=True)
    feature_df = feature_df.dropna(subset=["lag_7", "rolling_mean_7", "rolling_std_7"]).reset_index(drop=True)
    return feature_df
