import os

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import joblib
import matplotlib
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    model = joblib.load("model.pkl")
    df = pd.read_csv("data/processed/features.csv")
    X = df.drop(columns=["label"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        values_for_ai_class = shap_values[1]
    elif len(shap_values.shape) == 3:
        values_for_ai_class = shap_values[:, :, 1]
    else:
        values_for_ai_class = shap_values

    shap.summary_plot(values_for_ai_class, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    print("Saved shap_summary.png")


if __name__ == "__main__":
    main()
