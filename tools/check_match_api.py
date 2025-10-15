import requests
import json

URL = "http://127.0.0.1:8000/match"

payload = {
    "left": ["apple iphone 12", "brown leather boots"],
    "right": ["iphone 12 by apple", "waterproof hiking boots"],
    "min_score": 70.0,
}

r = requests.post(URL, json=payload)
print(r.status_code)
try:
    print(json.dumps(r.json(), indent=2))
except Exception:
    print(r.text)
