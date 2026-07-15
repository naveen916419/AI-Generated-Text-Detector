import pandas as pd
from transformers import pipeline


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


def main():
    generator = pipeline("text-generation", model="gpt2")
    generator.tokenizer.pad_token = generator.tokenizer.eos_token
    topics = TOPICS
    try:
        topics = pd.read_csv("data/raw/human_text.csv")["topic"].dropna().tolist()
    except FileNotFoundError:
        pass

    rows = []

    for topic in topics:
        prompt = f"Write a short informative paragraph about {topic}."
        output = generator(
            prompt,
            max_new_tokens=120,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.9,
            pad_token_id=generator.tokenizer.eos_token_id,
            clean_up_tokenization_spaces=False,
        )
        text = output[0]["generated_text"]
        rows.append({"text": text, "topic": topic, "label": "ai"})
        print(f"Generated: {topic}")

    df = pd.DataFrame(rows)
    df.to_csv("data/raw/ai_text_gpt2.csv", index=False)
    print(f"Saved {len(df)} AI samples")


if __name__ == "__main__":
    main()
