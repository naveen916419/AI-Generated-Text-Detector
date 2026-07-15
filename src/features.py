import numpy as np
import spacy
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


nlp = spacy.load("en_core_web_sm")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()


def sentence_perplexity(sentence):
    ids = tokenizer.encode(sentence, return_tensors="pt")
    if ids.shape[1] < 2:
        return None

    with torch.no_grad():
        loss = model(ids, labels=ids).loss

    return torch.exp(loss).item()


def extract_features(text):
    sentences = sent_tokenize(text)
    perplexities = [
        sentence_perplexity(sentence)
        for sentence in sentences
        if len(sentence.split()) > 2
    ]
    perplexities = [value for value in perplexities if value is not None]

    avg_perplexity = float(np.mean(perplexities)) if perplexities else 0
    burstiness = float(np.std(perplexities)) if len(perplexities) > 1 else 0

    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    pos_counts = {pos: pos_tags.count(pos) for pos in set(pos_tags)}
    total = sum(pos_counts.values()) or 1
    pos_probs = [count / total for count in pos_counts.values()]
    pos_entropy = float(-sum(prob * np.log2(prob) for prob in pos_probs if prob > 0))

    sent_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    sentence_len_var = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0

    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]
    word_count = len(words)
    type_token_ratio = len(set(words)) / len(words) if words else 0
    avg_sentence_length = word_count / len(sentences) if sentences else 0
    long_word_ratio = sum(1 for word in words if len(word) >= 7) / word_count if words else 0
    punctuation_ratio = sum(1 for char in text if char in ",;:!?") / len(text) if text else 0

    function_words = {"the", "of", "and", "a", "to", "in", "is", "that", "it", "was"}
    function_word_freq = (
        sum(1 for word in words if word in function_words) / len(words)
        if words
        else 0
    )

    return {
        "avg_perplexity": avg_perplexity,
        "burstiness": burstiness,
        "pos_entropy": pos_entropy,
        "sentence_len_var": sentence_len_var,
        "type_token_ratio": type_token_ratio,
        "function_word_freq": function_word_freq,
        "word_count": float(word_count),
        "avg_sentence_length": float(avg_sentence_length),
        "long_word_ratio": float(long_word_ratio),
        "punctuation_ratio": float(punctuation_ratio),
    }
