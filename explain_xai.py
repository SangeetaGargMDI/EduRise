
"""
This AI utilities uses SHAP. If SHAP is not installed, it falls back to simple feature importances.
"""

from pathlib import Path
import joblib
import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

MODELS_DIR = Path("models")

def load_model():
    bundle = joblib.load(MODELS_DIR / "edurise_rf.pkl")
    return bundle["model"], bundle["feature_names"]

def explain_instance(instance: np.ndarray):
    #for single instance it return feature_name, importance pairs list."""
    model, feature_names = load_model()
    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(instance)
        #Mean absolute SHAP value across classes
        importance = np.mean(np.abs(shap_values), axis=0).flatten()
        return list(zip(feature_names, importance))
    else:
        importances = model.feature_importances_
        return list(zip(feature_names, importances))

if __name__ == "__main__":
    model, feature_names = load_model()
    print("Loaded model with", len(feature_names), "features.")
