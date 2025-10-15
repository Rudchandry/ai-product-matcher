from fastapi.testclient import TestClient
from app.app import app
import json

client = TestClient(app)

payload = {
  "left": [
    "25 Gbps Direct Attach Cable (1m)",
    "Fortinet SFP+ Module (FN-TRAN-SFP+GC)",
    "Ubiquiti 10G SFP+ to 10/5/2.5GbE RJ45 Module",
    "Ubiquiti SFP+ Module (UACC-OM-SM-10G)",
    "Required Installation Materials",
    "Ubiquiti Networks - UniFi U7 Pro Max Tri-Band Wi-Fi 7 (Access Point)",
    "Fydelia Splash Page - Customer Enterprise Subscription"
  ],
  "right": [
    "Ubiquiti Networks -  UBU7PROMAX - UniFi U7 Pro Max Tri-Band Wi-Fi 7 Access Point",
    "Ubiquiti - UACC-OM-SM-10G-D-2 - 10 Gbps Single-Mode Optical Module (2-Pack)",
    "Ubiquiti - UACC-CM-RJ45-MG - SFP / SFP+ to RJ45 Adapter (10G)    ",
    "Ubiquiti - UACC-CM-RJ45-MG - SFP / SFP+ to RJ45 Adapter (10G)    ",
    "25G Direct Attach Cable 1M",
    "10GE copper SFP+ RJ45 Transceiver (30m range)\n",
    "Monoprice- 11274  - Cat6 Utp Cable,1 Ft - orange ",
    "New Splash Page Onboarding Subscription"
  ],
  "min_score": 70.0
}

r = client.post('/match', json=payload)
print(r.status_code)
print(json.dumps(r.json(), indent=2))
