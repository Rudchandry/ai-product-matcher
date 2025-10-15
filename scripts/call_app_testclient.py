import json
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)
print(json.dumps(client.get('/health').json(), indent=2))
resp = client.post('/predict', json={"pair": {"text_a": "Hello world", "text_b": "Hello"}})
print(json.dumps(resp.json(), indent=2))
