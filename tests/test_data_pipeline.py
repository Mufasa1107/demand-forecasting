import pandas as pd

from src.data.generate_dummy_data import generate_sales_dataset
from src.features.build_features import build_feature_frame


def test_generate_sales_dataset_returns_expected_shape_and_columns():
    df = generate_sales_dataset(
        regions=["North"],
        stores_per_region=2,
        categories_per_store=2,
        skus_per_category=2,
        periods=60,
        seed=42,
    )

    assert not df.empty
    assert set(
        [
            "date",
            "region",
            "store_id",
            "category",
            "sku",
            "demand",
            "promo_flag",
            "price",
        ]
    ).issubset(df.columns)
    assert df["sku"].nunique() == 4
    assert df["date"].dt.freq is not None or df["date"].nunique() > 0


def test_feature_engineering_adds_lag_and_calendar_features():
    df = generate_sales_dataset(
        regions=["North"],
        stores_per_region=1,
        categories_per_store=1,
        skus_per_category=2,
        periods=90,
        seed=7,
    )

    feature_df = build_feature_frame(df, horizon=7)

    expected_columns = {
        "day_of_week",
        "month",
        "is_weekend",
        "promo_flag",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_7",
    }
    assert expected_columns.issubset(feature_df.columns)
    assert feature_df["lag_7"].notna().all()
    assert feature_df["rolling_mean_7"].notna().all()
