import pandas as pd

from features import extract_features


def main():
    df = pd.read_csv("data/processed/labeled_dataset.csv")
    feature_rows = []

    for index, row in df.iterrows():
        features = extract_features(row["text"])
        features["label"] = row["label"]
        feature_rows.append(features)
        print(f"Processed {index + 1}/{len(df)}")

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv("data/processed/features.csv", index=False)
    print(feature_df.head())


if __name__ == "__main__":
    main()
