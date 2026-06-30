import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from demand_forecasting.config import DEFAULT_TARGET


def train_forecast_model(df: pd.DataFrame) -> dict:
    feature_columns = [col for col in df.columns if col not in {DEFAULT_TARGET, "date", "product_id"}]
    X = df[feature_columns]
    y = df[DEFAULT_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": rmse,
        "r2": r2_score(y_test, predictions),
    }
    return {"model": model, "metrics": metrics, "feature_columns": feature_columns}
