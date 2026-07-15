import streamlit as st
from pathlib import Path

from demand_forecasting.pipelines.forecasting_pipeline import run_forecasting_pipeline
from demand_forecasting.utils.logger import get_logger

logger = get_logger("streamlit_app", log_file="logs/app.log")

st.set_page_config(page_title="Ecommerce Demand Forecasting", layout="centered")

st.title("Ecommerce Demand Forecasting")
st.write("Upload a CSV with columns: date, product_id, demand")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    temp_path = Path("/tmp") / uploaded_file.name
    temp_path.write_bytes(uploaded_file.getvalue())
    logger.info("Uploaded file %s", uploaded_file.name)
    st.success(f"Loaded {uploaded_file.name}")

    with st.spinner("Running forecasting pipeline..."):
        result = run_forecasting_pipeline(str(temp_path), output_dir=Path("results"))

    st.subheader("Metrics")
    st.json(result["metrics"])

    st.caption(f"Data shape: {result['data_shape']}")
    st.caption(f"Feature shape: {result['feature_shape']}")
else:
    st.info("Please upload a CSV file to start.")
