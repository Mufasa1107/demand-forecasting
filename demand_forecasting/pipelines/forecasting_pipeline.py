from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from demand_forecasting.config import DEFAULT_TARGET
from demand_forecasting.data.ingestion import load_sales_data
from demand_forecasting.features.build_features import build_features
from demand_forecasting.models.train_model import train_forecast_model
from demand_forecasting.utils.logger import get_logger

logger = get_logger("forecasting_pipeline", log_file="logs/forecasting.log")


def run_forecasting_pipeline(data_path: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    logger.info("Loading sales data from %s", data_path)
    df = load_sales_data(data_path)
    if DEFAULT_TARGET not in df.columns:
        logger.warning("Target column '%s' missing; using fallback values", DEFAULT_TARGET)
        df[DEFAULT_TARGET] = 1

    logger.info("Building features")
    feature_df = build_features(df)
    logger.info("Training forecasting model")
    training_result = train_forecast_model(feature_df)

    output_dir = Path(output_dir or "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = training_result["metrics"]
    logger.info("Pipeline completed with metrics: %s", metrics)
    return {
        "data_shape": df.shape,
        "feature_shape": feature_df.shape,
        "metrics": metrics,
        "output_dir": str(output_dir),
    }
