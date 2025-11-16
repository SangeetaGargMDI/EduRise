
"""
Streamlit dashboard for EduRise (simple and intuitive).
To run:
    streamlit run dashboard_app.py
"""

import pandas as pd
from pathlib import Path
import streamlit as st
import joblib

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

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

def main():
    st.title("EduRise – Enrolment Risk Dashboard")
    st.write("Predict, explain and engage to improve enrolment in government schools.")

    df = load_data()
    bundle = load_model_bundle()
    if df.empty or bundle is None:
        st.stop()

    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # Select school
    if "school_name" in df.columns:
        school_name = st.selectbox("Select school", sorted(df["school_name"].unique()))
        row = df[df["school_name"] == school_name].iloc[0]
    else:
        st.warning("No 'school_name' column found, using first row as demo.")
        row = df.iloc[0]
        school_name = row.get("school_id", "Demo School")

    st.subheader(f"School: {school_name}")
    st.write("Row data preview:", row.to_dict())

    # Build feature vector
    X = row[feature_names].values.reshape(1, -1)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    st.markdown(f"**Predicted enrolment bucket:** {pred}")
    st.markdown("**Class probabilities:**")
    prob_table = pd.DataFrame({"class": model.classes_, "probability": proba})
    st.table(prob_table)

    st.subheader("Explanation (XAI)")
    st.info("Connect this to explain_xai.explain_instance for per-school drivers.")

    st.subheader("AI-generated recommendation")
    st.info("Connect this to generate_reports.build_prompt and your LLM backend.")

if __name__ == "__main__":
    main()
