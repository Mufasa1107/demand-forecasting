import pandas as pd

from demand_forecasting.config import DEFAULT_DATE_COLUMN, DEFAULT_ITEM_COLUMN, DEFAULT_LAGS, DEFAULT_TARGET


def build_features(df: pd.DataFrame, lags: tuple[int, ...] = DEFAULT_LAGS) -> pd.DataFrame:
    if df.empty:
        return df

    feature_df = df.copy()
    feature_df = feature_df.sort_values([DEFAULT_ITEM_COLUMN, DEFAULT_DATE_COLUMN]).reset_index(drop=True)

    for lag in lags:
        feature_df[f"lag_{lag}"] = feature_df.groupby(DEFAULT_ITEM_COLUMN)[DEFAULT_TARGET].shift(lag).fillna(0)

    feature_df["day_of_week"] = feature_df[DEFAULT_DATE_COLUMN].dt.dayofweek
    feature_df["month"] = feature_df[DEFAULT_DATE_COLUMN].dt.month
    feature_df["is_weekend"] = feature_df["day_of_week"].isin([5, 6]).astype(int)
    feature_df = feature_df.reset_index(drop=True)
    return feature_df
