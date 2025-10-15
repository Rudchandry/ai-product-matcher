from fastapi.testclient import TestClient
from app.app import app

print("Calling GET /health and POST /predict using TestClient context manager")
payload = {"pair": {"text_a": "Hello world", "text_b": "Hello world"}}
with TestClient(app) as client:
    try:
        r = client.get("/health")
        print("GET /health ->", r.status_code)
        print(r.json())
    except Exception as e:
        print("Health request failed:", e)

    try:
        r = client.post("/predict", json=payload)
        print("POST /predict ->", r.status_code)
        print(r.json())
    except Exception as e:
        print("Predict request failed:", e)
