import pandas as pd


def main():
    human = pd.read_csv("data/raw/human_text.csv")
    ai = pd.read_csv("data/raw/ai_text_gpt2.csv")

    df = pd.concat([human, ai], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("data/processed/labeled_dataset.csv", index=False)

    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
