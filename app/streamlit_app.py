import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))
from features import extract_features  # noqa: E402


@st.cache_resource
def load_model():
    return joblib.load(ROOT_DIR / "model.pkl")


model = load_model()

st.title("AI-Generated Text Detector")
st.write("Paste any text below to check if it is likely human-written or AI-generated.")

text = st.text_area("Enter text:", height=200)

if st.button("Analyze") and text.strip():
    with st.spinner("Analyzing text... this may take a few seconds."):
        features = extract_features(text)
        X = pd.DataFrame([features])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

    label = "AI-Generated" if prediction == 1 else "Human-Written"
    confidence = probability if prediction == 1 else 1 - probability

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.1%}")
    st.subheader("Feature Breakdown")
    st.json(features)
