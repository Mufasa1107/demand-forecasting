from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def generate_sales_dataset(
    regions: List[str],
    stores_per_region: int,
    categories_per_store: int,
    skus_per_category: int,
    periods: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic hierarchical retail sales dataset."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for region in regions:
        for store_idx in range(stores_per_region):
            store_id = f"{region.lower()}_store_{store_idx + 1}"
            for category_idx in range(categories_per_store):
                category = f"cat_{category_idx + 1}"
                for sku_idx in range(skus_per_category):
                    sku = f"{category}_sku_{sku_idx + 1}"
                    base_level = 1200 + rng.integers(0, 400)
                    seasonality = np.sin(np.arange(periods) / 7.0) * 70
                    trend = np.linspace(0, 80, periods)
                    promo = rng.binomial(1, 0.12, size=periods).astype(int)
                    price = 10 + rng.uniform(0, 3, size=periods)
                    for t in range(periods):
                        date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=t)
                        demand = (
                            base_level
                            + trend[t]
                            + seasonality[t]
                            + (promo[t] * 150)
                            + (store_idx * 40)
                            + (category_idx * 20)
                            + rng.normal(0, 25)
                        )
                        demand = max(int(round(demand * (1.0 + (price[t] - 10) * 0.02))), 0)
                        rows.append(
                            {
                                "date": date,
                                "region": region,
                                "store_id": store_id,
                                "category": category,
                                "sku": sku,
                                "demand": demand,
                                "promo_flag": promo[t],
                                "price": round(float(price[t]), 2),
                            }
                        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df
