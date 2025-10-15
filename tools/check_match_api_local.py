from fastapi.testclient import TestClient
from app.app import app
import json

client = TestClient(app)

payload = {
    "left": ["apple iphone 12", "brown leather boots"],
    "right": ["iphone 12 by apple", "waterproof hiking boots"],
    "min_score": 70.0,
}

r = client.post("/match", json=payload)
print(r.status_code)
try:
    print(json.dumps(r.json(), indent=2))
except Exception:
    print(r.text)
