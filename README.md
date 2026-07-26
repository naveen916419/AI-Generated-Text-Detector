# AI-Generated Text Detector

Beginner-friendly NLP project that detects whether text is likely human-written or AI-generated using stylometric and perplexity-based features.

## Project Flow

1. Collect human-written text.
2. Generate AI-written text on the same topics.
3. Combine both into a labeled dataset.
4. Extract linguistic features.
5. Train and evaluate a classifier.
6. Explain predictions with SHAP.
7. Run a Streamlit demo.

## Current Status

The project now trains on a balanced starter dataset of 62 samples:

- 31 human-written Wikipedia summaries
- 31 GPT-2 generated paragraphs on the same topics

Current held-out test result:

- Accuracy: 100%
- ROC-AUC: 1.0
- Test size: 13 samples

Note: this is a strong starter result, but the test set is still small. For a resume-ready version, the dataset should be expanded further and tested on text from an unseen AI model.

## Run Locally

```powershell
venv\Scripts\activate
streamlit run app\streamlit_app.py --server.fileWatcherType none
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository and push this project, including `model.pkl`.
   Do not commit the local `venv/` folder.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**
   and select the GitHub repository and branch.
3. Set the main file path to `app/streamlit_app.py`, then click **Deploy**.

The first launch downloads the GPT-2 model used to calculate perplexity, so it may
take longer than normal. The app then caches the loaded models for later requests.
