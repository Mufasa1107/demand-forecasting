import tempfile
import unittest
from pathlib import Path

import pandas as pd

from demand_forecasting.pipelines.forecasting_pipeline import run_forecasting_pipeline


class ForecastingPipelineTest(unittest.TestCase):
    def test_pipeline_runs_with_sample_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sales.csv"
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]),
                    "product_id": [1, 1, 1, 1, 1, 1],
                    "demand": [10, 11, 13, 12, 14, 15],
                }
            )
            df.to_csv(csv_path, index=False)

            result = run_forecasting_pipeline(str(csv_path), output_dir=Path(tmp_dir) / "results")

            self.assertIn("metrics", result)
            self.assertIn("mae", result["metrics"])
            self.assertTrue(Path(result["output_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
