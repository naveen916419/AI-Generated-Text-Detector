import time
from urllib.parse import quote

import pandas as pd
import requests


TOPICS = [
    "Climate change",
    "Artificial intelligence",
    "History of India",
    "Space exploration",
    "Healthy eating",
    "Cricket",
    "Renewable energy",
    "Ancient civilizations",
    "Social media",
    "Mental health",
    "Travel",
    "Cybersecurity",
    "Wildlife conservation",
    "Online education",
    "Startup company",
    "Electric vehicles",
    "Public health",
    "Financial literacy",
    "Ocean pollution",
    "Urban planning",
    "Classical music",
    "Machine learning",
    "Renewable agriculture",
    "Blockchain",
    "Remote work",
    "Vaccination",
    "Digital privacy",
    "E-commerce",
    "World War II",
    "Yoga",
    "Robotics",
    "Water scarcity",
    "Film industry",
    "Cloud computing",
    "Sustainable fashion",
    "Nutrition",
    "Democracy",
    "Virtual reality",
    "Entrepreneurship",
    "Disaster management",
    "Quantum computing",
    "Public transportation",
    "Marine biology",
    "Data science",
    "Education reform",
    "Renewable resources",
    "Globalization",
    "Personal finance",
    "Human rights",
    "Electric power",
]


def fetch_wikipedia_summary(topic):
    title = quote(topic.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    response = requests.get(
        url,
        headers={"User-Agent": "ai-text-detector-beginner-project/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("extract", "")


def main():
    rows = []

    for topic in TOPICS:
        try:
            summary = fetch_wikipedia_summary(topic)
            if not summary:
                raise ValueError("No summary text returned")
            rows.append({"text": summary, "topic": topic, "label": "human"})
            print(f"Collected: {topic}")
        except Exception as exc:
            print(f"Skipped {topic}: {exc}")
        time.sleep(1)

    df = pd.DataFrame(rows)
    df.to_csv("data/raw/human_text.csv", index=False)
    print(f"Saved {len(df)} human samples")


if __name__ == "__main__":
    main()
