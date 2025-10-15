from app.app import app
from fastapi.testclient import TestClient

c = TestClient(app)

r = c.get('/health')
print('health', r.status_code, r.text)

payload = {
    'left': ['apple iphone 12'],
    'right': ['Apple iPhone 12'],
    'min_score': 0
}

r = c.post('/match', json=payload)
print('match', r.status_code, r.text)
