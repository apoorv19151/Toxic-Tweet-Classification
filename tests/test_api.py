from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

def test_home():
    response = client.get("/")

    assert response.status_code == 200

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_empty_text():
    response = client.post(
        "/predict",
        json={"text": "   ", "model": "distilbert"}
    )

    assert response.status_code == 400

def test_missing_text():
    response = client.post(
        "/predict",
        json={"model": "distilbert"}
    )

    assert response.status_code == 422

def test_prediction():
    with patch("app.predict", return_value="not_cyberbullying"):
        response = client.post(
            "/predict",
            json={
                "text": "Have a wonderful day!",
                "model": "distilbert"
            }
        )

    assert response.status_code == 200
    assert response.json() == {
        "prediction": "not_cyberbullying"
    }
