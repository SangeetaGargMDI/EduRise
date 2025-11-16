"""
Simple Streamlit dashboard for EduRise.

To run:
    streamlit run dashboard_app.py
"""

import pandas as pd
from pathlib import Path
import streamlit as st
import joblib
import openai  # must be installed
from generate_reports import build_prompt

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

# ----------------------------------------------------------------------
# LOADERS
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    path = PROCESSED_DATA_DIR / "modelling_table.csv"
    if not path.exists():
        st.error("Processed data not found. Please run data_preprocessing.py first.")
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_resource
def load_model_bundle():
    model_path = MODELS_DIR / "edurise_rf.pkl"
    if not model_path.exists():
        st.error("Model file not found. Please run model_training.py first.")
        return None
    return joblib.load(model_path)

# ----------------------------------------------------------------------
# DRIVER EXTRACTION
# ----------------------------------------------------------------------
def get_top_drivers(model, feature_names, X, top_k=5):
    """Extract top drivers using feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        pairs = list(zip(feature_names, importances))
        sorted_drivers = sorted(pairs, key=lambda x: x[1], reverse=True)
        return sorted_drivers[:top_k]
    else:
        return [("N/A", 0.0)]

# ----------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------
def main():
    st.title("EduRise – Enrolment Risk Dashboard")
    st.write("Predict, explain and engage to improve enrolment in government schools.")

    df = load_data()
    bundle = load_model_bundle()
    if df.empty or bundle is None:
        st.stop()

    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # -----------------------------
    # Select school
    # -----------------------------
    if "school_name" in df.columns:
        school_name = st.selectbox(
            "Select school",
            sorted(df["school_name"].unique())
        )
        row = df[df["school_name"] == school_name].iloc[0]
    else:
        st.warning("No 'school_name' column found — using first row.")
        row = df.iloc[0]
        school_name = row.get("school_id", "Demo School")

    st.subheader(f"School: {school_name}")
    st.write("Row data preview:", row.to_dict())

    # -----------------------------
    # Prediction
    # -----------------------------
    X = row[feature_names].values.reshape(1, -1)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    st.markdown(f"### Predicted enrolment bucket: **{pred}**")

    prob_table = pd.DataFrame({"class": model.classes_, "probability": proba})
    st.table(prob_table)

    # -----------------------------
    # EXTRACT DRIVERS FOR PROMPT
    # -----------------------------
    drivers = get_top_drivers(model, feature_names, X)

    st.subheader("Top Drivers")
    st.write(drivers)

    # -----------------------------
    # AI-GENERATED RECOMMENDATIONS
    # -----------------------------
    st.subheader("AI-generated recommendations")

    # Extract optional location field
    location = row.get("location", "")  

    # -----------------------------
    # Prepare LLM prompt
    # -----------------------------
    prompt = build_prompt(
        school_name=school_name,
        location=location,
        risk_bucket=pred,
        drivers=drivers,
    )

    st.text_area("LLM prompt (editable)", prompt, height=300)

    # -----------------------------
    # OPENAI CLIENT
    # -----------------------------
    try:
        client = openai.OpenAI()
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return

    # -----------------------------
    # Generate Recommendations
    # -----------------------------
    if st.button("Generate Recommendations"):
        with st.spinner("Generating recommendations..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                output = response.choices[0].message.content
                st.write(output)

            except Exception as e:
                st.error(f"LLM Error: {e}")


# ----------------------------------------------------------------------
# ENTRY
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
