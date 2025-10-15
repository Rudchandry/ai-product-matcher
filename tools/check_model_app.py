from fastapi.testclient import TestClient
import json

from app import model_app

client = TestClient(model_app.app)


def main():
    print("GET /health")
    r = client.get("/health")
    try:
        print(r.status_code, json.dumps(r.json(), indent=2))
    except Exception:
        print(r.status_code, r.text)

    print("POST /predict (sample)")
    payload = {"pair": {"text_a": "hello", "text_b": "world"}}
    r2 = client.post("/predict", json=payload)
    try:
        print(r2.status_code, json.dumps(r2.json(), indent=2))
    except Exception:
        print(r2.status_code, r2.text)


if __name__ == "__main__":
    main()
