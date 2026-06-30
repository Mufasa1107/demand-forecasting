from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_TARGET = "demand"
DEFAULT_DATE_COLUMN = "date"
DEFAULT_ITEM_COLUMN = "product_id"
DEFAULT_LAGS = (1, 7, 14)
