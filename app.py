"""
Kerala Real Estate Price Predictor
-----------------------------------
Streamlit app that serves the trained Keras regression model produced by the
accompanying notebook (project.ipynb).

Expected artifacts (produced by the notebook, must sit next to this file,
or be uploaded via the sidebar):
    - best_model.keras
    - scaler.joblib
    - encoders.joblib          (dict: column -> fitted LabelEncoder)
    - features.joblib          (list of feature column names, in model order)
    - default_features.joblib  (dict: column -> default/median/mode value)

Run with:
    streamlit run app.py
"""

import os
import io
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kerala Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
)

ARTIFACT_NAMES = {
    "model": "best_model.keras",
    "scaler": "scaler.joblib",
    "encoders": "encoders.joblib",
    "features": "features.joblib",
    "defaults": "default_features.joblib",
}


# ----------------------------------------------------------------------------
# Artifact loading (cached so the model/scaler are only loaded once)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_keras_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


@st.cache_resource(show_spinner=False)
def load_joblib_artifact(path):
    return joblib.load(path)


def find_artifact(name, uploaded_bytes_map):
    """Look for an artifact first among files the user uploaded in the
    sidebar, otherwise fall back to a file living next to this script."""
    if name in uploaded_bytes_map:
        return uploaded_bytes_map[name]
    if os.path.exists(name):
        return name
    return None


def load_all_artifacts(uploaded_bytes_map):
    missing = []
    loaded = {}

    model_path = find_artifact(ARTIFACT_NAMES["model"], uploaded_bytes_map)
    scaler_path = find_artifact(ARTIFACT_NAMES["scaler"], uploaded_bytes_map)
    encoders_path = find_artifact(ARTIFACT_NAMES["encoders"], uploaded_bytes_map)
    features_path = find_artifact(ARTIFACT_NAMES["features"], uploaded_bytes_map)
    defaults_path = find_artifact(ARTIFACT_NAMES["defaults"], uploaded_bytes_map)

    for key, path in [
        ("model", model_path),
        ("scaler", scaler_path),
        ("encoders", encoders_path),
        ("features", features_path),
        ("defaults", defaults_path),
    ]:
        if path is None:
            missing.append(ARTIFACT_NAMES[key])

    if missing:
        return None, missing

    loaded["model"] = load_keras_model(model_path)
    loaded["scaler"] = load_joblib_artifact(scaler_path)
    loaded["encoders"] = load_joblib_artifact(encoders_path)
    loaded["features"] = load_joblib_artifact(features_path)
    loaded["defaults"] = load_joblib_artifact(defaults_path)
    return loaded, []


# ----------------------------------------------------------------------------
# Sidebar: artifact upload (fallback if files aren't bundled with the app)
# ----------------------------------------------------------------------------
st.sidebar.title("🏠 Kerala Real Estate")
st.sidebar.caption(
    "Neural-network price predictor trained in `project.ipynb`."
)

with st.sidebar.expander("⚙️ Model artifacts", expanded=False):
    st.write(
        "The app looks for these files in the same folder as `app.py`. "
        "If they're missing, upload them here."
    )
    uploaded_files = st.file_uploader(
        "Upload artifacts",
        type=["keras", "joblib"],
        accept_multiple_files=True,
    )

uploaded_bytes_map = {}
if uploaded_files:
    tmp_dir = ".uploaded_artifacts"
    os.makedirs(tmp_dir, exist_ok=True)
    for f in uploaded_files:
        dest = os.path.join(tmp_dir, f.name)
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        uploaded_bytes_map[f.name] = dest

artifacts, missing = load_all_artifacts(uploaded_bytes_map)

page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict Price", "📊 Data Exploration", "📁 Batch Prediction", "ℹ️ About"],
)

# ----------------------------------------------------------------------------
# Helper: build a single-row DataFrame from form inputs, encode + scale it
# ----------------------------------------------------------------------------
def prepare_input(raw_values: dict, features, encoders, scaler):
    row = pd.DataFrame([raw_values], columns=features)
    for col, le in encoders.items():
        if col in row.columns:
            val = str(row.at[0, col])
            # Guard against unseen categories at inference time
            if val not in list(le.classes_):
                val = le.classes_[0]
            row[col] = le.transform([val])
    row = row[features]  # keep model's expected column order
    scaled = scaler.transform(row)
    return scaled


def predict_price(raw_values: dict, artifacts):
    scaled = prepare_input(
        raw_values,
        artifacts["features"],
        artifacts["encoders"],
        artifacts["scaler"],
    )
    pred = artifacts["model"].predict(scaled, verbose=0)
    return float(np.ravel(pred)[0])


# ----------------------------------------------------------------------------
# PAGE: Predict Price
# ----------------------------------------------------------------------------
if page == "🔮 Predict Price":
    st.title("🔮 Predict Property Price")

    if missing:
        st.error(
            "Missing required artifact(s): " + ", ".join(missing) +
            "\n\nPlace them next to `app.py`, or upload them via the "
            "sidebar (⚙️ Model artifacts)."
        )
        st.stop()

    features = artifacts["features"]
    encoders = artifacts["encoders"]
    defaults = artifacts["defaults"]

    st.write(
        "Fill in the property details below and click **Predict** to get "
        "an estimated market price."
    )

    with st.form("prediction_form"):
        cols = st.columns(2)
        raw_values = {}

        for i, feat in enumerate(features):
            target_col = cols[i % 2]
            default_val = defaults.get(feat)

            with target_col:
                if feat in encoders:
                    options = list(encoders[feat].classes_)
                    default_index = (
                        options.index(default_val)
                        if default_val in options
                        else 0
                    )
                    raw_values[feat] = st.selectbox(
                        feat.replace("_", " ").title(),
                        options=options,
                        index=default_index,
                    )
                else:
                    default_num = float(default_val) if default_val is not None else 0.0
                    step = 1.0 if float(default_num).is_integer() else 0.01
                    raw_values[feat] = st.number_input(
                        feat.replace("_", " ").title(),
                        value=default_num,
                        step=step,
                        format="%.4f",
                    )

        submitted = st.form_submit_button("💰 Predict Price", use_container_width=True)

    if submitted:
        with st.spinner("Running the model..."):
            try:
                price = predict_price(raw_values, artifacts)
                st.success(f"### Estimated Price: ₹ {price:,.2f}")

                # Simple confidence framing using +/- a small band around the point estimate
                low, high = price * 0.9, price * 1.1
                st.caption(f"Typical range: ₹{low:,.0f} – ₹{high:,.0f}")

                with st.expander("Show input sent to the model"):
                    st.json(raw_values)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.session_state.setdefault("history", [])
    if submitted:
        st.session_state["history"].append({**raw_values, "predicted_price": price})

    if st.session_state.get("history"):
        st.divider()
        st.subheader("🕘 Prediction History (this session)")
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, use_container_width=True)
        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download history as CSV",
            data=csv_bytes,
            file_name="prediction_history.csv",
            mime="text/csv",
        )


# ----------------------------------------------------------------------------
# PAGE: Data Exploration (mirrors the EDA done in the notebook)
# ----------------------------------------------------------------------------
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    st.write(
        "Upload the raw dataset (e.g. `kerala_real_estate_geo_rebuilt.csv`) "
        "to reproduce the exploratory analysis from the notebook: outlier "
        "detection, correlation heatmap, and price relationships."
    )

    data_file = st.file_uploader("Upload CSV dataset", type=["csv"])

    if data_file is not None:
        df = pd.read_csv(data_file)

        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isna().sum().sum()))

        with st.expander("Column info"):
            buf = io.StringIO()
            df.info(buf=buf)
            st.text(buf.getvalue())

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if num_cols:
            import matplotlib.pyplot as plt
            import seaborn as sns

            st.subheader("Outlier Detection (IQR method)")
            outlier_col = st.selectbox("Choose a numeric column", num_cols)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df[outlier_col], ax=ax)
            ax.set_title(f"Boxplot of {outlier_col}")
            st.pyplot(fig)

            q1, q3 = df[outlier_col].quantile(0.25), df[outlier_col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = df[(df[outlier_col] < lower) | (df[outlier_col] > upper)].shape[0]
            st.caption(f"IQR bounds: [{lower:,.2f}, {upper:,.2f}] — {n_outliers} outliers found")

            st.subheader("Correlation Heatmap")
            fig2, ax2 = plt.subplots(figsize=(min(1.2 * len(num_cols), 16), 8))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)

            if "price" in df.columns:
                st.subheader("Price Distribution")
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                sns.histplot(df["price"], bins=30, kde=True, ax=ax3)
                st.pyplot(fig3)

                st.subheader("Price vs. Another Feature")
                other_cols = [c for c in num_cols if c != "price"]
                if other_cols:
                    x_col = st.selectbox("Feature to compare against price", other_cols)
                    fig4, ax4 = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(x=df[x_col], y=df["price"], ax=ax4)
                    ax4.set_title(f"Price vs {x_col}")
                    st.pyplot(fig4)

                cat_cols = df.select_dtypes(include="object").columns.tolist()
                if cat_cols:
                    st.subheader("Average Price by Category")
                    cat_col = st.selectbox("Categorical column", cat_cols)
                    avg = (
                        df.groupby(cat_col)["price"]
                        .mean()
                        .sort_values(ascending=False)
                    )
                    st.bar_chart(avg)
        else:
            st.info("No numeric columns detected for outlier/correlation analysis.")
    else:
        st.info("Upload a CSV file above to explore the data.")


# ----------------------------------------------------------------------------
# PAGE: Batch Prediction
# ----------------------------------------------------------------------------
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction")

    if missing:
        st.error(
            "Missing required artifact(s): " + ", ".join(missing) +
            "\n\nPlace them next to `app.py`, or upload them via the "
            "sidebar (⚙️ Model artifacts)."
        )
        st.stop()

    features = artifacts["features"]
    encoders = artifacts["encoders"]

    st.write(
        "Upload a CSV containing the following columns (order doesn't "
        "matter, extra columns are ignored):"
    )
    st.code(", ".join(features))

    batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            missing_cols = [c for c in features if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Uploaded file is missing columns: {missing_cols}")
            else:
                work_df = batch_df[features].copy()

                for col, le in encoders.items():
                    if col in work_df.columns:
                        work_df[col] = work_df[col].astype(str).apply(
                            lambda v: v if v in list(le.classes_) else le.classes_[0]
                        )
                        work_df[col] = le.transform(work_df[col])

                scaled = artifacts["scaler"].transform(work_df[features])
                preds = artifacts["model"].predict(scaled, verbose=0).ravel()

                result_df = batch_df.copy()
                result_df["predicted_price"] = preds

                st.success(f"Predicted prices for {len(result_df)} rows.")
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_bytes,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")
    else:
        st.info("Upload a CSV file above to run predictions on multiple rows at once.")


# ----------------------------------------------------------------------------
# PAGE: About
# ----------------------------------------------------------------------------
else:
    st.title("ℹ️ About this app")
    st.markdown(
        """
This app serves the deep-learning regression model trained in
**`project.ipynb`** for predicting Kerala real-estate prices.

**Pipeline (from the notebook):**
1. Load `kerala_real_estate_geo_rebuilt.csv`, drop `City/Place` and `date`.
2. Impute missing numeric values with the column median.
3. Detect & winsorize outliers using the IQR method.
4. Encode categorical columns with `LabelEncoder`.
5. Scale features with `StandardScaler`.
6. Train a fully-connected neural network:
   `Dense(256) → BatchNorm → Dropout → Dense(128) → Dropout → Dense(64) → Dense(32) → Dense(1)`
   with Adam optimizer, MSE loss, and early stopping on validation loss.
7. Evaluate with R², MAE, MSE, and RMSE.

**This app adds:**
- An interactive prediction form built dynamically from the saved feature
  list, encoders, and default values — so it stays in sync with retraining.
- A data-exploration page reproducing the notebook's outlier, correlation,
  and price-distribution charts on any uploaded dataset.
- Batch prediction from CSV with a downloadable results file.
- A session-based prediction history you can export.
- A sidebar uploader so the model artifacts can be supplied at runtime if
  they aren't bundled alongside the app.
        """
    )

    if artifacts is not None:
        st.subheader("Loaded feature order")
        st.code(", ".join(artifacts["features"]))
        st.subheader("Encoded (categorical) columns")
        st.code(", ".join(artifacts["encoders"].keys()) or "None")