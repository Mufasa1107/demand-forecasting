import argparse
import json
from pathlib import Path

from demand_forecasting.pipelines.forecasting_pipeline import run_forecasting_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ecommerce demand forecasting pipeline")
    parser.add_argument("--data", required=True, help="Path to a CSV file containing sales history")
    parser.add_argument("--output-dir", default="results", help="Directory for generated artifacts")
    args = parser.parse_args()

    result = run_forecasting_pipeline(args.data, output_dir=Path(args.output_dir))
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
