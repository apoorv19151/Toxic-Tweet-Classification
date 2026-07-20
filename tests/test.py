from src.inference import predict

VALID_LABELS = {
    "religion",
    "age",
    "ethnicity",
    "gender",
    "other_cyberbullying",
    "not_cyberbullying",
}

MODELS = [
    "lr_tfidf",
    "bilstm_glove",
    "bilstm_word2vec",
    "distilbert"
]

TEXT = (
    "Just got the new iPhone and I am absolutely in love "
    "with the camera quality! Best purchase ever."
)

for model in MODELS:
    prediction = predict(
        TEXT,
        model_name=model,
    )

    assert prediction in VALID_LABELS

    print(f"{model}: {prediction}")
