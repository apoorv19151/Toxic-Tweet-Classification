import pytest

from src.inference import predict, SUPPORTED_MODELS

VALID_LABELS = {
    "religion",
    "age",
    "ethnicity",
    "gender",
    "other_cyberbullying",
    "not_cyberbullying",
}

TEXT = (
    "Just got the new iPhone and I am absolutely in love "
    "with the camera quality! Best purchase ever."
)

@pytest.mark.parametrize(
    "model_name",
    sorted(SUPPORTED_MODELS)
)

def test_model_prediction(model_name):
    prediction = predict(
        TEXT,
        model_name=model_name,
    )

    assert prediction in VALID_LABELS

def test_invalid_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        predict(
            TEXT,
            model_name="invalid_model",
        )
