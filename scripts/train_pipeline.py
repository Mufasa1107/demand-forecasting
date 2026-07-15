from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.generate_dummy_data import generate_sales_dataset
from src.models.reconcile import reconcile_hierarchical_forecasts
from src.models.train_model import train_lightgbm_forecasts
from src.utils.config import ProjectConfig


def main() -> None:
    cfg = ProjectConfig()
    df = generate_sales_dataset(
        regions=list(cfg.regions),
        stores_per_region=cfg.stores_per_region,
        categories_per_store=cfg.categories_per_store,
        skus_per_category=cfg.skus_per_category,
        periods=cfg.periods,
        seed=cfg.seed,
    )
    predictions = train_lightgbm_forecasts(df, horizon=cfg.horizon)
    reconciled = reconcile_hierarchical_forecasts(predictions)

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    reconciled.to_csv(output_dir / "hierarchical_forecasts.csv", index=False)
    print(f"Saved {len(reconciled)} rows to {output_dir / 'hierarchical_forecasts.csv'}")


if __name__ == "__main__":
    main()
