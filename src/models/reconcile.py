from __future__ import annotations

import pandas as pd


def reconcile_hierarchical_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Create a coherent hierarchy by aggregating bottom-level forecasts upward."""
    base_df = df.copy()
    if "sku" in base_df.columns and "sku_id" not in base_df.columns:
        base_df = base_df.rename(columns={"sku": "sku_id"})
    if "region" not in base_df.columns or "store_id" not in base_df.columns or "category" not in base_df.columns:
        raise KeyError("Forecasts must include region, store_id, and category before reconciliation")
    required_columns = ["date", "region", "store_id", "category", "sku_id", "prediction"]
    missing = [col for col in required_columns if col not in base_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for reconciliation: {missing}")
    base_df = base_df[required_columns].copy()

    groups = [
        ["region"],
        ["region", "store_id"],
        ["region", "store_id", "category"],
        ["region", "store_id", "category", "sku_id"],
    ]

    rows = []
    for level in groups:
        level_name = "|".join(level)
        aggregated = (
            base_df.groupby(level + ["date"])["prediction"].sum().reset_index(name="prediction")
        )
        aggregated["level"] = level_name
        aggregated["node"] = aggregated[level].astype(str).agg(lambda row: "|".join(row), axis=1)
        rows.append(aggregated[["date", "level", "node", "prediction"]])

    reconciled = pd.concat(rows, ignore_index=True)
    reconciled["date"] = pd.to_datetime(reconciled["date"])
    return reconciled
