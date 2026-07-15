import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def main():
    df = pd.read_csv("data/processed/features.csv")
    X = df.drop(columns=["label"])
    y = (df["label"] == "ai").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    }

    best_name = None
    best_model = None
    best_auc = -1

    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        probabilities = candidate.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
        print(f"{name} ROC-AUC: {auc:.3f}")
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = candidate

    model = best_model

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))

    joblib.dump(model, "model.pkl")
    print("Saved model.pkl")


if __name__ == "__main__":
    main()
