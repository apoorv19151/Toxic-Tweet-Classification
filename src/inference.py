import argparse

from pathlib import Path
import joblib

import torch

from src.models import BiLSTMClassifier
from src.preprocessing import (
    clean_text_tfidf,
    clean_text_lstm,
    clean_text_transformer,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models"

MAX_LEN = 64
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.5
NUM_CLASSES = 6

HF_MODEL_ID = "singhapoorv82/cyberbullying-distilbert"
HF_MODEL_REVISION = "04257272d9a652dd1efb4ed50310752ad36607be"

DEFAULT_MODEL = "lr_tfidf"
SUPPORTED_MODELS = {
    "lr_tfidf",
    "bilstm_glove",
    "bilstm_word2vec",
    "distilbert"
}

_lr_model = None
_tfidf_vectorizer = None
_label_mapping = None
_bilstm_glove = None
_bilstm_word2vec = None
_word2idx = None
_distilbert_model = None
_distilbert_tokenizer = None

def _load_classical():
    global _lr_model
    global _tfidf_vectorizer

    if _lr_model is None:
        _lr_model = joblib.load(MODEL_PATH / "classical/lr_tfidf.pkl")

    if _tfidf_vectorizer is None:
        _tfidf_vectorizer = joblib.load(MODEL_PATH / "classical/tfidf_vectorizer.pkl")

    return _lr_model, _tfidf_vectorizer, _load_labels()

def _load_bilstm(model_name: str):
    global _bilstm_glove
    global _bilstm_word2vec
    global _word2idx

    if _word2idx is None:
        _word2idx = joblib.load(MODEL_PATH / "bilstm/word2idx.pkl")

    if model_name == "bilstm_glove":
        if _bilstm_glove is None:

            _bilstm_glove = BiLSTMClassifier(
                vocab_size=len(_word2idx),
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
                num_classes=NUM_CLASSES,
                padding_idx=_word2idx["<PAD>"],
            )

            state_dict = torch.load(MODEL_PATH / "bilstm/bilstm_glove.pth", map_location=DEVICE)

            _bilstm_glove.load_state_dict(state_dict)

            _bilstm_glove.to(DEVICE)

            _bilstm_glove.eval()

        return _bilstm_glove, _word2idx

    if model_name == "bilstm_word2vec":
        if _bilstm_word2vec is None:

            _bilstm_word2vec = BiLSTMClassifier(
                vocab_size=len(_word2idx),
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
                num_classes=NUM_CLASSES,
                padding_idx=_word2idx["<PAD>"],
            )

            state_dict = torch.load(MODEL_PATH / "bilstm/bilstm_word2vec.pth", map_location=DEVICE)

            _bilstm_word2vec.load_state_dict(state_dict)

            _bilstm_word2vec.to(DEVICE)

            _bilstm_word2vec.eval()

        return _bilstm_word2vec, _word2idx

    raise ValueError(f"Unknown model {model_name}.")

def _load_distilbert():
    global _distilbert_model
    global _distilbert_tokenizer

    if _distilbert_tokenizer is None:
        _distilbert_tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_ID,
            revision=HF_MODEL_REVISION
        )
    
    if _distilbert_model is None:
        _distilbert_model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_ID,
            revision=HF_MODEL_REVISION
        )

        _distilbert_model.to(DEVICE)
        _distilbert_model.eval()

    return _distilbert_model, _distilbert_tokenizer

def _load_labels():
    global _label_mapping

    if _label_mapping is None:
        label2id = joblib.load(
            MODEL_PATH / "classical/label_mapping.pkl"
        )
        _label_mapping = {
            value: key for key, value in label2id.items()
        }

    return _label_mapping

def _text_to_ids(text: str, word2idx: dict) -> list[int]:
    tokens = text.split()
    return [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]

def _pad_sequence(sequence: list[int], pad_idx: int) -> list[int]:
    if len(sequence) > MAX_LEN:
        return sequence[:MAX_LEN]
    
    return sequence + [pad_idx] * ((MAX_LEN) - len(sequence))

def _predict_bilstm(text: str, *, model_name: str) -> str:
    model, word2idx = _load_bilstm(model_name)

    label_mapping = _load_labels()

    cleaned_text = clean_text_lstm(text)

    sequence = _text_to_ids(cleaned_text, word2idx)
    sequence = _pad_sequence(sequence, word2idx["<PAD>"])

    sequence = torch.tensor([sequence], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits = model(sequence)
        prediction = torch.argmax(logits, dim=1).item()

    return label_mapping[prediction]

def _predict_classical(text: str) -> str:
    model, vectorizer, label_mapping = _load_classical()
    
    cleaned_text = clean_text_tfidf(text)
    features = vectorizer.transform([cleaned_text])

    prediction = model.predict(features)[0]

    return label_mapping[prediction]

def _predict_distilbert(text: str) -> str:
    model, tokenizer = _load_distilbert()
    
    label_mapping = _load_labels()

    cleaned_text = clean_text_transformer(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1,
        ).item()

    return label_mapping[prediction]

def predict(text: str, *, model_name: str | None = None):
    if model_name is None:
        model_name = DEFAULT_MODEL

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models: {SUPPORTED_MODELS}"
        )

    if model_name == "lr_tfidf":
        return _predict_classical(text)

    if model_name == "bilstm_glove":
        return _predict_bilstm(text, model_name="bilstm_glove")

    if model_name == "bilstm_word2vec":
        return _predict_bilstm(text, model_name="bilstm_word2vec")

    if model_name == "distilbert":
        return _predict_distilbert(text)

    raise ValueError("Unknown model.")

def main():
    parser = argparse.ArgumentParser(
        description="Cyberbullying Tweet Classifier",
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text to classify.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=sorted(SUPPORTED_MODELS),
        help="Model to use.",
    )

    args = parser.parse_args()

    prediction = predict(
        args.text,
        model_name=args.model,
    )

    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
