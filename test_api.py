import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    api_client = TestClient(app)
    return api_client

def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the App for this project"}

def test_post_below50k(client):
    r = client.post("/", json={
        "age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "maritalStatus": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States",
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_above50k(client):
    r = client.post("/", json={
        "age": 61,
        "workclass": "Private",
        "education": "Doctorate",
        "maritalStatus": "Separated",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 60,
        "nativeCountry": "United-States",
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}