from fastapi.testclient import TestClient
from app.app import app

with TestClient(app) as client:
    r = client.get('/')
    print('GET / ->', r.status_code, r.json())
    r = client.get('/items/42?q=abc')
    print('GET /items/42?q=abc ->', r.status_code, r.json())
