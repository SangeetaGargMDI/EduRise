
"""
Model training module for EduRise.

Trains a simple tree-based model to predict enrolment_change_pct bucket.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_modelling_table() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "modelling_table.csv"
    return pd.read_csv(path)

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Define a simple 3-class target based on enrolment change percentage.
    change = df["enrolment_change_pct"].fillna(0)
    bins = [-1e9, -5, 5, 1e9]
    labels = ["decline", "stable", "growth"]
    df["target_bucket"] = pd.cut(change, bins=bins, labels=labels)
    return df

def train_model(random_state: int = 42) -> RandomForestClassifier:
    df = load_modelling_table()
    df = create_target(df)
    # Drop rows with missing target
    df = df.dropna(subset=["target_bucket"])
    y = df["target_bucket"]
    # Very naive feature selection: drop identifiers and target columns
    drop_cols = {"school_id", "school_name", "enrolment_current", "enrolment_next", "enrolment_change_pct", "target_bucket"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    model_path = MODELS_DIR / "edurise_rf.pkl"
    joblib.dump({"model": clf, "feature_names": list(X.columns)}, model_path)
    print(f"Saved model to {model_path}")
    return clf

if __name__ == "__main__":
    train_model()
