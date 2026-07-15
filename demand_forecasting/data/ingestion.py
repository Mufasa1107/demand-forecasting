from pathlib import Path
from typing import Union

import pandas as pd

from demand_forecasting.config import DEFAULT_DATE_COLUMN, DEFAULT_ITEM_COLUMN


def load_sales_data(
    file_path: Union[str, Path],
    date_column: str = DEFAULT_DATE_COLUMN,
    item_column: str = DEFAULT_ITEM_COLUMN,
) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, parse_dates=[date_column])
    if date_column not in df.columns:
        raise ValueError(f"Expected date column '{date_column}' in data")
    if item_column not in df.columns:
        raise ValueError(f"Expected item column '{item_column}' in data")
    return df
